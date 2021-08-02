from core.dataset.data_zoo import get_data_by_name


if __name__ == '__main__':

    print("[INFO] Loading Dataset...")

    train, val, test = get_data_by_name("image_caption")

    print(train, val, test)

    train, val, test = get_data_by_name("image_folder", )

    print(train, val, test)

    print("[INFO] Done.")
