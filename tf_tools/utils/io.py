import os

def save_in_file(content, path, mode='w'):
    file_dir = path[:path.rfind('/')]

    create_folder(file_dir)

    with open(path, 'w') as f:
        f.write(results)

def create_folder(directory):
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
