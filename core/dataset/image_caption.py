import os
import pandas
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def process_csv(scv_path):
    # 1.读取CSV文件
    csv_frame = pandas.read_csv(scv_path)
    # 2.将表格的第2列，根据第0列建立索引。(按列号索引)
    csv_frame = csv_frame.groupby(csv_frame.columns[0])[csv_frame.columns[2]]

    # 2.将表格中名为comment的列，根据名为image_name的列建立索引。(按列名索引)
    #   但不建议在这里使用按列名索引。
    # csv_frame = csv_frame.groupby('image_name')['comment']

    # 3.根据索引重新建立表头为["image_name", "comment"]的表格，被合并的单元格内使用"#"分隔
    csv_frame = csv_frame.apply('#'.join).reset_index()

    # 4.转换成列表(按列号索引) 建议使用按列号索引
    image_name = csv_frame[csv_frame.columns[0]].to_list()
    image_label = csv_frame[csv_frame.columns[1]].to_list()

    # 4.转换成列表(按列名索引) 不建议使用按列名索引 结果同上
    # image_name = csv_frame["image_name"].to_list()
    # image_label = csv_frame["comment"].to_list()

    # 5.合并两个列表
    data_merge = list(zip(image_name, image_label))

    return data_merge


class ImageCaption(Dataset):
    def __init__(self, data_root, transform):
        super(ImageCaption, self).__init__()
        self.data_root = data_root
        self.data_list = process_csv(os.path.join(data_root, "train_labels.csv"))
        self.transform = transform

    def __getitem__(self, idx):
        # 拼接图片完整路径
        image_path = os.path.join(self.data_root, "images", str(self.data_list[idx][0]))
        # 读取路径对应的图片
        image = Image.open(image_path)
        # 通过transform做变换（对应数据预处理）
        image_tensor = self.transform(image)
        # 返回 图片(已处理) 和 标签(未处理, 这里截取长度为100用于测试)
        return image_tensor, str(self.data_list[idx][1])[:100]

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # 测试CSV读取是否正确
    print("---"*20, "process_csv", "---"*20)
    data = process_csv("../../dataset/demo/train_labels.csv")
    print(data[0])
    print(data[1])

    # 测试数据集读取是否正确
    print("---" * 20, "ImageCaption", "---" * 20)
    my_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    my_dataset = ImageCaption("../../dataset/demo/", my_transform)
    print(my_dataset)
    my_data_loader = DataLoader(my_dataset, batch_size=8, shuffle=True, num_workers=1)

    for image, label in my_data_loader:
        print(image.shape)
        print(label)
