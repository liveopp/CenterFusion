from dataset.nuscenes import NuScenes

dataset_factory = {
  # 'custom': CustomDataset,
  # 'coco': COCO,
  # 'kitti': KITTI,
  # 'coco_hp': COCOHP,
  # 'mot': MOT,
  'nuscenes': NuScenes,
  # 'crowdhuman': CrowdHuman,
  # 'kitti_tracking': KITTITracking,
}


def get_dataset(dataset):
  return dataset_factory[dataset]
