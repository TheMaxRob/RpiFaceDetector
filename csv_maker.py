import os

base_path = "./friend_faces"
output_csv = "./friend_faces.csv"

# Check if directory exists
if not os.path.exists(base_path):
    print(f"Error: Directory '{base_path}' does not exist")
    print(f"Current working directory: {os.getcwd()}")
    exit(1)

try:
    with open(output_csv, "w") as f:
        label = 0
        for subject_dir in sorted(os.listdir(base_path)):
            subject_path = os.path.join(base_path, subject_dir)
            if os.path.isdir(subject_path):
                for image in sorted(os.listdir(subject_path)):
                    image_path = os.path.join(subject_path, image)
                    f.write(f"{image_path};{label}\n")
                label += 1
    print(f"CSV file created: {output_csv}")
except Exception as e:
    print(f"An error occurred: {e}")