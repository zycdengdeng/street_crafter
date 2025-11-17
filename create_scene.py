from street_gaussian.models.street_gaussian_model import StreetGaussianModel
from street_gaussian.datasets.dataset import Dataset
from street_gaussian.pointcloud_processor import getPointCloudProcessor
from street_gaussian.models.scene import Scene
from street_gaussian.config import cfg
from video_diffusion.sample_condition import VideoDiffusionModel


def create_scene():
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.metadata)
    if (cfg.mode == 'train' and cfg.diffusion.use_diffusion) or cfg.mode == 'diffusion':
        print("Diffusion model loaded")
        video_diffusion = VideoDiffusionModel(
            config_path=cfg.diffusion.config_path,
            ckpt_path=cfg.diffusion.ckpt_path,
            height=cfg.diffusion.height,
            width=cfg.diffusion.width
        )
    else:
        video_diffusion = None

    if cfg.mode == 'train' or cfg.mode == 'diffusion':
        pointcloud_processor = getPointCloudProcessor()
    else:
        pointcloud_processor = None

    scene = Scene(
        gaussians=gaussians,
        dataset=dataset,
        pointcloud_processor=pointcloud_processor,
        diffusion=video_diffusion,
    )

    return scene
