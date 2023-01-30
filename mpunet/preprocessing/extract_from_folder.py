import os
import shutil

if __name__ == '__main__':

    root = '/Users/px/Downloads/testdata 3'
    for i in os.listdir(root):
        sub_dir = os.path.join(root, i)

        if not os.path.isdir(sub_dir):
            continue

        file_name = os.listdir(sub_dir)[0]

        before_path = os.path.join(sub_dir, file_name)
        after_path = os.path.join(root, file_name)

        shutil.move(before_path, after_path)
        shutil.rmtree(sub_dir)

