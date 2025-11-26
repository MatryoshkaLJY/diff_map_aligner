import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys
from collections import defaultdict

def largest_connected_component_uf(points):
    if not points:
        return []
    
    parent = {p: p for p in points}
    size = {p: 1 for p in points}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa == pb:
            return
        if size[pa] < size[pb]:
            pa, pb = pb, pa
        parent[pb] = pa
        size[pa] += size[pb]
    
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    point_set = set(points)
    
    for x, y in points:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) in point_set:
                union((x,y), (nx,ny))
    
    # 找到最大的根
    root_to_points = defaultdict(list)
    for p in points:
        root = find(p)
        root_to_points[root].append(p)
    
    largest_comp = max(root_to_points.values(), key=len)
    return sorted(largest_comp)

def batch_paste_with_preview_and_correct_tiff(
    ref_path,
    folder_path,
    output_tif="批量贴图结果_多层.tif",
    sample_step=16,
    diff_thresh=128
):
    
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print("参考图读取失败！")
        return
    h, w = ref_img.shape[:2]
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # SIFT + FLANN
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.01, edgeThreshold=10)
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
    matcher = cv2.FlannBasedMatcher({"algorithm":1, "trees":5}, {"checks":50})

    # 用于最终预览（BGR）
    preview = ref_img.copy()
    preview0 = ref_img.copy()

    # 用于保存的图层列表（必须是 RGB 顺序！）
    layers_pil = []
    borders_info = []  # (color_bgr, name)
    diff_points_all = []  # 存储所有差异点用于图例

    # 添加底图
    layers_pil.append(Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)))

    print(f"开始批量匹配（采样步长={sample_step}px，差异阈值={diff_thresh}）...")

    for idx, file in enumerate(sorted(Path(folder_path).iterdir())):
        if file.suffix.lower() not in {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}:
            continue
        print(f"处理: {file.name}", end="")

        img2 = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
        if img2 is None:
            print(" → 读取失败")
            continue

        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape)==3 else img2.mean(axis=2).astype(np.uint8)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        if des2 is None:
            print(" → 无特征点")
            continue

        matches = matcher.knnMatch(des_ref, des2, k=2)
        good = [m for m,n in matches if m.distance < 0.82 * n.distance]
        if len(good) < 12:
            print(f" → 匹配太少({len(good)})")
            continue

        src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        if H is None or mask.sum() < 10:
            print(" → RANSAC失败")
            continue

        print(f" → 成功! 内点:{mask.sum()}")

        # 透视变换
        border_mode = cv2.BORDER_TRANSPARENT if img2.shape[2] == 4 else cv2.BORDER_CONSTANT
        warped = cv2.warpPerspective(img2, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode)

        # 生成粗掩码（贴图有内容区域）
        if warped.shape[2] == 4:
            content_mask = warped[:,:,3] > 30
        else:
            content_mask = cv2.warpPerspective((cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) > 20).astype(np.uint8), H, (w, h)) > 0

        # 融合到临时预览图
        for c in range(3):
            preview0[:,:,c] = np.where(content_mask, 
                (warped[:,:,c]).astype(np.uint8),
                ref_img[:,:,c])


        # 半透明融合到预览图
        alpha = 0.7
        for c in range(3):
            preview[:,:,c] = np.where(content_mask, 
                (warped[:,:,c] * alpha + preview[:,:,c] * (1-alpha)).astype(np.uint8),
                preview[:,:,c])
            
        
        # 画彩色边框（表示整体区域）
        contours, _ = cv2.findContours((content_mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_color = tuple(int(x) for x in np.random.randint(80, 200, 3))
        cv2.drawContours(preview, contours, -1, area_color, thickness=4)
        borders_info.append((area_color, file.stem))

        # ==================== 新增：SIFT采样差异检测 ====================
        # 在贴图内容区域内，每隔 sample_step 像素采样
        # 更高效的间隔采样：先下采样掩码，再找非零点并映射回原坐标
        sub = content_mask[::sample_step, ::sample_step]
        ys_sub, xs_sub = np.nonzero(sub)
        if len(ys_sub) == 0:
            sample_points = []
        else:
            # 映射回原图坐标（使用采样格左上角作为点）
            sample_points = [(int(x * sample_step), int(y * sample_step)) for y, x in zip(ys_sub, xs_sub)]
        print(f" | 采样点数: {len(sample_points)}", end="")

        # 计算采样点的 SIFT 描述子
        kp_warped, des_warped = sift.compute(
            cv2.cvtColor(preview0, cv2.COLOR_BGR2GRAY),
            [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=20) for p in sample_points]
        )

        kp_ref_sample, des_ref_sample = sift.compute(
            ref_gray,
            [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=20) for p in sample_points]
        )

        diff_points = []
        if des_warped is not None and des_ref_sample is not None:
            for i, (d1, d2) in enumerate(zip(des_warped, des_ref_sample)):
                dist = np.linalg.norm(d1 - d2)
                if dist > diff_thresh:
                    x, y = int(sample_points[i][0]), int(sample_points[i][1])
                    diff_points.append((x, y))

        #             diff_points.append((x//sample_step, y//sample_step))
        # # 使用并查集找最大连通组件，过滤孤立点
        # diff_points = largest_connected_component_uf(diff_points)
        # # 映射回原坐标
        # diff_points = [(x * sample_step, y * sample_step) for x, y in diff_points]

        print(f" → 发现 {len(diff_points)} 个显著差异点")
        # 在预览图上标记差异点（红色圆点）
        for x, y in diff_points:
            cv2.circle(preview, (x, y), 3, (0, 0, 255), -1)

        diff_points_all.extend(diff_points)

        if diff_points:  # 只有当这张图有显著差异点时才生成
            # Step 1: 创建空白掩码 (h, w)
            diff_mask = np.zeros((h, w), dtype=np.uint8)

            # Step 2: 把所有差异点（已映射回原坐标）画成填充圆（半径稍大一点避免稀疏）
            for x, y in diff_points:
                cv2.circle(diff_mask, (x, y), radius=25, color=255, thickness=-1)

            # Step 3: 闭运算填补小空洞 + 高斯模糊羽化边缘（核心！）
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            # diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            diff_mask = cv2.GaussianBlur(diff_mask, (51, 51), sigmaX=25, sigmaY=25)  # 强羽化

            # Step 4: 应用到 warped 图层（保留透明通道）
            if warped.shape[2] == 4:
                # 有 alpha 通道的图
                highlighted = warped.copy()
                alpha_new = (highlighted[:,:,3].astype(np.float32) * (diff_mask.astype(np.float32) / 255.0)).astype(np.uint8)
                highlighted[:,:,3] = alpha_new
            else:
                # 无 alpha 的图，重建 RGBA
                highlighted = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
                highlighted[:,:,3] = diff_mask  # 直接用羽化后的灰度图作为 alpha

            # 转为 PIL 并加入图层列表（插在原图层后面）
            pil_highlight = Image.fromarray(cv2.cvtColor(highlighted, cv2.COLOR_BGRA2RGBA)).convert("RGBA")
            layers_pil.append(pil_highlight)

            # 可选：在预览图上也叠加半透明红色高亮（更醒目）
            overlay = preview.copy()
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
            overlay[:,:,3] = np.clip(overlay[:,:,3].astype(int) + (diff_mask.astype(int) * 0.4).astype(int), 0, 255).astype(np.uint8)
            preview = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)

            print(f" → 已生成羽化差异高亮图层（{len(diff_points)}个种子点 → 实心+羽化区域）")
        else:
            print(" → 无显著差异，跳过高亮图层生成")

        # ==================== 原有正常贴图图层（保持不变）===================
        if warped.shape[2] == 4:
            rgba = cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA)
            pil_layer = Image.fromarray(rgba).convert("RGBA")
        else:
            rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            pil_layer = Image.fromarray(rgb).convert("RGBA")
            alpha_channel = np.where(content_mask, 255, 0).astype(np.uint8)
            pil_layer.putalpha(Image.fromarray(alpha_channel))
        layers_pil.append(pil_layer)

    # ==================== 显示对比 ====================
    plt.figure(figsize=(24, 12))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    plt.title("Reference Image", fontsize=16)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    plt.title(f"Found {len(diff_points_all)} significant difference points\n"
              f"Colored border = subimage region | Red dot = local difference", fontsize=16)
    plt.axis('off')

    # 图例
    for i, (color_bgr, name) in enumerate(borders_info):
        color_rgb = tuple(c/255 for c in color_bgr[::-1])
        plt.plot(0,0, color=color_rgb, linewidth=6, label=f"Region {i+1}: {name}")
    if diff_points_all:
        plt.plot(0,0, 'o', color='red', markersize=10, label=f"Difference Points ×{len(diff_points_all)}")
    if borders_info or diff_points_all:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.show()

    # ==================== 保存多页 TIFF ====================
    layers_pil[0].save(
        output_tif,
        save_all=True,
        append_images=layers_pil[1:],
        compression="tiff_deflate",
        tiffinfo={270: "SIFT Batch Paste + Local Difference Detection", 305: "OpenCV + Python"}
    )
    print(f"\n多页 TIFF 保存成功：{output_tif}")
    print(f"总共检测到 {len(diff_points_all)} 个局部显著差异点")

# ======================== 运行 ========================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python xxx.py <path_to_reference_image> <path_to_overlays_folder>")
        print("example: python xxx.py base.jpg overlays/")
        sys.exit(1)

    batch_paste_with_preview_and_correct_tiff(
        sys.argv[1],
        sys.argv[2],
        "output.tif",
        sample_step=16,      # 可调：越大越快，越小越密集
        diff_thresh=128     # 可调：越大越严格（推荐 0.5~0.7）
    )