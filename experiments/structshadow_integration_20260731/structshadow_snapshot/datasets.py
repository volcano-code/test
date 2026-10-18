"""数据加载模块 - 直接来自 mesh8.1.7"""
import cv2
import numpy as np
import torch


def get_props(img, img_size=(256, 256)):
    """图像预处理：居中和缩放"""
    rows, cols = img_size
    h0, w0 = (int(rows*0.7), int(cols*0.7))
    cx0, cy0 = (img_size[0]//2, img_size[1]//2)
    
    conts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conts = sorted(conts, key=lambda ctr: cv2.contourArea(ctr), reverse=True)
    holes = len(conts) - 1
    
    if len(conts) == 0:
        return img, holes
    
    cnt = conts[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cx1 = x + w/2
    cy1 = y + h/2
    
    sx = w0 * 1.0 / max(w, 1)
    sy = h0 * 1.0 / max(h, 1)
    tx = cx0 * (1 - sx) + (cx0 - cx1)
    ty = cy0 * (1 - sy) + (cy0 - cy1)
    
    M = np.float32([[sx, 0, tx], [0, sy, ty]])
    img = cv2.warpAffine(img, M, img_size)
    
    return img, holes


def load_data_from_list(files, mirror_mode, img_size, device, num_views, debug_mode=False):
    """从文件列表加载数据"""
    assert len(files) == num_views, f"文件数量({len(files)})与视图数量({num_views})不匹配"
    
    images = []
    img_tensors = []
    
    for file_ in files:
        print(f"  Loading: {file_}")
        _, img = cv2.threshold(cv2.imread(file_, 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = img.astype(np.uint8)
        img = cv2.resize(img, img_size)
        img, holes = get_props(img, img_size)
        
        if debug_mode:
            cv2.imshow("img", img)
            cv2.waitKey(0)
        
        images.append(img)
    
    if mirror_mode:
        for file_ in files:
            _, img = cv2.threshold(cv2.imread(file_, 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = img.astype(np.uint8)
            img = cv2.resize(img, img_size)
            img = cv2.flip(img, 1)
            img, holes = get_props(img, img_size)
            images.append(img)
    
    for img in images:
        img_tensor = torch.from_numpy(img / 255.0).float().to(device)
        img_tensors.append(img_tensor)
    
    return images, img_tensors
