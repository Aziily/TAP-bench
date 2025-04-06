import os

def get_sketch_data_path(path):
    # dataset_type, data_path, query_first, resize_to_256
    return {
        "tapvid_davis_first": ("davis", os.path.join(path, "tapvid_davis", "tapvid_davis.pkl"), True),
        "tapvid_davis_strided": ("davis", os.path.join(path, "tapvid", "tapvid_davis", "tapvid_davis.pkl"), False),
        "tapvid_kinetics_first": ("kinetics", os.path.join(path, "tapvid_kinetics"), True),
        "tapvid_rgb_stacking_first": ("stacking", os.path.join(path, "tapvid_rgb_stacking", "tapvid_rgb_stacking.pkl"), True),
        "tapvid_robotap_first": ("robotap", os.path.join(path, "tapvid_robotap"), True),
    }
    
def get_perturbed_data_path(path):
    # dataset_type, data_path, query_first, resize_to_256
    return {
        "tapvid_davis_first": ("davis", os.path.join(path, "tapvid_davis", "tapvid_davis.pkl"), True),
        "tapvid_davis_strided": ("davis", os.path.join(path, "tapvid", "tapvid_davis", "tapvid_davis.pkl"), False),
        "tapvid_kinetics_first": ("kinetics", os.path.join(path, "tapvid_kinetics"), True),
        "tapvid_rgb_stacking_first": ("stacking", os.path.join(path, "tapvid_rgb_stacking", "tapvid_rgb_stacking.pkl"), True),
        "tapvid_robotap_first": ("robotap", os.path.join(path, "tapvid_robotap"), True),
    }
    
def get_depth_root_from_data_root(data_root):
    if "kinetics" in data_root or "robotap" in data_root:
        return os.path.join(data_root, "video_depth_anything")
    return os.path.join("/", *data_root.split("/")[:-1], "video_depth_anything")