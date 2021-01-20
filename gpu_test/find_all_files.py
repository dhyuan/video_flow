import os

import shutil


def list_all_files(path_str):
    all_files = []
    for path, d, file_list in os.walk(path_str):
        print(file_list)
        print(len(all_files))
        for filename in file_list:
            file = os.path.join(path, filename)
            all_files.append(file)
    return all_files


def copy_and_rename_all_files_to(from_dir, target_dir):
    i = 0
    targets = []
    for file in files:
        name = os.path.basename(file)
        post_fix = os.path.splitext(file)[-1]
        target_file = "%s/%d%s" % (target_dir, i, post_fix)
        shutil.copy(file, target_file)
        targets.append(target_file)
        i += 1
    return targets


if __name__ == '__main__':
    files = list_all_files('./images/1000')
    result = copy_and_rename_all_files_to('./images/1000', './images/1001')
    print(result)

