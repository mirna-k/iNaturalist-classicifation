import os

def rename_images(folder_path, prefix='image', start_index=1):
    """
    Rename all image files in a folder with a specified prefix and sequential index.

    Args:
    - folder_path (str): Path to the folder containing the image files.
    - prefix (str): Prefix to be added to the new names (default is 'image').
    - start_index (int): Starting index for the sequential numbering (default is 1).
    """

    if not os.path.exists(folder_path):
        print("Folder path does not exist.")
        return

    files = os.listdir(folder_path)
    
    # Filter out only image files (you may want to extend this list for other image formats)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    index = start_index
    for old_name in image_files:
        _, ext = os.path.splitext(old_name)
        
        new_name = f"{prefix}_{index:03d}{ext}"
       
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
       
        os.rename(old_path, new_path)
        
        index += 1
