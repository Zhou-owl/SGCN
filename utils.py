import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)

def loc_pos(seq_):

    # seq_ [obs_len pedsN 2]

    obs_len = seq_.shape[0]
    num_ped = seq_.shape[1]
    # 1D, [1:obs_len]
    pos_seq = np.arange(1, obs_len + 1)
    # expand to 3D (obs_len, 1, 1)
    pos_seq = pos_seq[:, np.newaxis, np.newaxis]
    # (obs_len, 2, 1)
    pos_seq = pos_seq.repeat(num_ped, axis=1)
    # (obs_len, num_ped, 3), result[x,:,0] = x+1
    result = np.concatenate((pos_seq, seq_), axis=-1)
    return result

def seq_to_graph(seq_, seq_rel, pos_enc=False):
    # (valid_peds_num_in_this_seq, 2, frame(8 or 12))
    # squeeze: remove dimension if is 1
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]
    # (frameNum, pedsNum, 2)
    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        # frame S (pedsNum,2)
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            # V(frameID, pedID, 2)
            V[s, h, :] = step_rel[h]

    if pos_enc:
        # obs_len, num_ped, 3), result[x, :, 0] = x + 1
        V = loc_pos(V)

    return torch.from_numpy(V).type(torch.float)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    # polyfit returns a set of values represented the correlation between two number series if full=True
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1] #NOTE this syntax
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    # match tab
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            # strip: remove space or newline characters at the beginning and the end of the string
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    # a little different from np.array
    return np.asarray(data)

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        for path in all_files:
            data = read_file(path, delim)
            # unique: remove duplicates in list
            frames = np.unique(data[:, 0]).tolist() # all frame id
            frame_data = []
            for frame in frames:
                # (exp) frame_data[0]: all data in frame 0
                frame_data.append(data[frame == data[:, 0], :]) # NOTE the use of this syntax
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                # concatenate all data in one trajectory(20 frames) by row (keep col unchanged)
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                # all peds id in current trajectory
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                # (peds_num, 2(feature), obs+pred len)
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    # all data for each ped in current seq (frame_id, ped_id, pos_X, pos_Y)
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    # the first frame of this ped's appearance (only considering this seq, maximum 20)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    # the ped should appear in all frames of one seq
                    if pad_end - pad_front != self.seq_len:
                        continue
                    # transpose this matrix, remain all 20 rows and cal[2:], from(20, 4) to (2,20)
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) # NOTE this syntax
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative, (2,20), the relative position of each frame and this previous frame
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    # ipdb.set_trace()
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    # rel_curr_ped_seq[:, 1:] = \
                    #     curr_ped_seq[:, 1:] - np.reshape(curr_ped_seq[:, 0], (2,1))
                    _idx = num_peds_considered
                    # curr_seq(all_peds_num, 2, 20), data in curr_seq[valid_peds_num:all_peds_num] is zero
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    # 1 if the ped's trajectory is non-linear
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                # have traversed all peds in curr seq
                if num_peds_considered > min_ped:
                    # +=: concat to the end
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    #seq_list(seq_num, valid_ped_in_this_seq, 2, 20)
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        # concatenate to 3 dimension (valid_ped_in_all_seq, 2, 20)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        # ones(all_peds_num, 20)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        # 1D, all peds' linear nature (0->linear 1->nonlinear)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        # (valid_ped_in_all_seq, 2, 8)
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        # (valid_ped_in_all_seq, 2, 12)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        # cumsum: return the cumulative sum of the elements along the gicen axis;
        # cum_start idx: the start cum id of ped in each seq (start from 0)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        # start cum id, end cum id in each seq
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs
        self.v_obs = []
        self.v_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]
            # obs_traj: (all_valid_peds_num, 2, frames(8)), pass valid peds of each seq into <seqtograph>
            # v: (obj_len, peds_num, feature(3))m feature[0]=frame_id+1, feature[1] and future[2] = rel_pos
            v_= seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], True)
            self.v_obs.append(v_.clone())
            v_= seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], False)
            self.v_pred.append(v_.clone())

        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.v_pred[index]
        ]
        return out
