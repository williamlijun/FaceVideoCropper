# 由于openface的处理可能出现失败的情况，为了保证图像数据与openface的输出结果保持一致性，需要为这两种数据建立一个映射连接
# 该代码会处理openface的输出结果，对产生的csv文件添加两个字段，一个 img_path 表示图片的相对路径，一个label表示这一张图片的类别
# 该代码应在当前目录下运行 即从./python/data

import os
import pandas as pd

def build_joint_csv(unstructured_path):
    dataset_name = os.path.basename(unstructured_path) # affectnet
    output_dir = os.path.join("dataset", dataset_name) # ./dataset/affectnet
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] 输出目录: {output_dir}")

    for label in os.listdir(unstructured_path):
        label_path = os.path.join(unstructured_path, label)
        if not os.path.isdir(label_path):
            continue

        print(f"[INFO] 处理类别: {label}")
        dfs = []

        for fname in os.listdir(label_path):
            if not fname.endswith('_of_details.txt'):
                continue

            txt_path = os.path.join(label_path, fname)
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            # 判断是否处理成功
            if len(lines) < 13:
                continue

            # 提取 image 文件名
            input_line = next((line for line in lines if line.startswith("Input:")), None)
            if input_line is None:
                continue

            image_name = os.path.basename(input_line.strip().split("Input:")[-1].strip())
            image_base = os.path.splitext(image_name)[0]
            image_rel_path = f"{label}/{image_name}"

            # 构造对应的 csv 路径
            csv_filename = f"{image_base}.csv"
            csv_path = os.path.join(label_path, csv_filename)
            if not os.path.exists(csv_path):
                continue

            try:
                df = pd.read_csv(csv_path)
                df['img_path'] = image_rel_path
                df['label'] = label
                dfs.append(df)
            except Exception as e:
                print(f"[ERROR] 读取失败: {csv_path} -> {e}")
                continue

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            out_path = os.path.join(output_dir, f"{label}.csv")
            merged_df.to_csv(out_path, index=False)
            print(f"[SUCCESS] 保存: {out_path}")
        else:
            print(f"[WARNING] 类别 {label} 无有效数据，跳过。")

# 示例调用方式
if __name__ == "__main__":
    build_joint_csv("./structured/affectnet")
