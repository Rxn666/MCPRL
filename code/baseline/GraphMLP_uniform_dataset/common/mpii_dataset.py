import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates

h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                  16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])


mpii_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70,  # Only used for visualization
    }
]

mpii_cameras_extrinsic_params = {
    'S0': [
        {
            'orientation': [0.0, 0.0, 0.0, 1.0],  # Identity quaternion
            'translation': [0.0, 0.0, 0.0],  # Zero translation
        },
    ]}
for i in range(200, 222):
    subject = f'S{i}'
    mpii_cameras_extrinsic_params[subject] = copy.deepcopy(mpii_cameras_extrinsic_params['S0'])



class MPIIDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=True):
        super().__init__(fps=25, skeleton=h36m_skeleton)  # Adjust the skeleton and fps as needed
        self.train_list = ['S200', 'S201', 'S202', 'S203', 'S204', 'S205', 'S206', 'S207', 'S208', 'S209',
                           'S210', 'S211', 'S212', 'S213', 'S214', 'S215']  # Adjust as needed
        self.test_list = ['S216', 'S217', 'S218', 'S219', 'S220', 'S221']  # Adjust as needed

        self._cameras = copy.deepcopy(mpii_cameras_extrinsic_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(mpii_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2

                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000

                cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion']))

        data = np.load(path, allow_pickle=True)['positions_3d'].item()

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras[subject],
                }

        if remove_static_joints:
            self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])  # Adjust as needed
            # Adjust parent joints as needed
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8

    def supports_semi_supervised(self):
        return False  # or False, depending on your case
