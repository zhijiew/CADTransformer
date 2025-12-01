import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg', type=str, default="config/hrnet48.yaml", help='experiment configure file name')
    parser.add_argument('--model', type=str, default="cadformer", choices=["cadformer", "dual_branch_cadformer"], help='model type')
    parser.add_argument('--val_only', action="store_true", help='flag to do evaluation on val set')
    parser.add_argument('--test_only', action="store_true", help='flag to do evaluation on test set')
    parser.add_argument('--data_root', type=str, default="/ssd1/zhiwen/projects/CADTransformer/data/floorplan_v1")
    parser.add_argument('--embed_backbone', type=str, default="hrnet48")
    parser.add_argument('--pretrained_model', type=str, default="./pretrained_models/HRNet_W48_C_ssld_pretrained.pth")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0, dest='local_rank')
    parser.add_argument('--log_step', type=int, default=100, help='steps for logging')
    parser.add_argument('--img_size', type=int, default=700, help='image size of rasterized image')
    parser.add_argument('--max_prim', type=int, default=12000, help='maximum primitive number for each batch')
    parser.add_argument('--load_ckpt', type=str, default='', help='load checkpoint')
    parser.add_argument('--resume_ckpt', type=str, default='', help='continue train while loading checkpoint')
    parser.add_argument('--log_dir', type=str, default='', help='logging directory')
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('--debug', action="store_true")

    parser.add_argument('--lambda_raster', type=float, default=1.0, help='weight for raster loss')
    parser.add_argument('--lambda_vector', type=float, default=1.0, help='weight for vector loss')
    parser.add_argument('--lambda_align', type=float, default=0.1, help='weight for alignment loss')
    parser.add_argument('--align_temperature', type=float, default=0.07, help='temperature for alignment loss')

    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args


__all__ = ["get_parser", "parse_args"]
