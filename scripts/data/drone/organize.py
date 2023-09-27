import os
import shutil
from pathlib import Path
import pyrootutils


def main():
    cwd = Path().resolve()
    rootdir = pyrootutils.setup_root(
        search_from=cwd,
        indicator=".project-root",
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=True,
    )  # root: root folder path of the directory

    print("Path for the root folder is:")
    print(rootdir)

    root_directory = os.path.join(rootdir, "datasets/Drone")
    hr_directory = os.path.join(rootdir, "datasets/DRONE/HR")
    lq_directory = os.path.join(rootdir, "datasets/DRONE/LQ")

    if not os.path.exists(hr_directory):
        print("making HR directory")
        os.makedirs(hr_directory)
    if not os.path.exists(lq_directory):
        print("making LR directory")
        os.makedirs(lq_directory)

    for root, dirs, files in os.walk(root_directory):
        path_parts = root.split(os.sep)
        if len(path_parts) >= 5 and path_parts[-2] in [
            "10",
            "20",
            "30",
            "40",
            "50",
            "70",
            "80",
            "100",
            "120",
            "140",
        ]:
            drone_id, mission_id, altitude, burst_number = (
                path_parts[-4],
                path_parts[-3],
                path_parts[-2],
                path_parts[-1],
            )
            for filename in files:
                lq_directory_altitude = os.path.join(lq_directory, f"{altitude}")
                hr_directory_altitude = os.path.join(hr_directory, f"{altitude}")
                if not os.path.exists(lq_directory_altitude):
                    print(f"making LR altitude directory: {lq_directory_altitude}")
                    os.makedirs(lq_directory_altitude)

                if not os.path.exists(hr_directory_altitude):
                    print(f"making LR altitude directory: {hr_directory_altitude}")
                    os.makedirs(hr_directory_altitude)

                if filename == "color_correction.png":
                    dest_path = os.path.join(
                        lq_directory_altitude,
                        f"{drone_id}_{mission_id}_{altitude}_{burst_number}.png",
                    )
                    print(f"Copying {filename} to {dest_path}")
                    shutil.copy(os.path.join(root, filename), dest_path)
                elif filename == "tele.png":
                    dest_path = os.path.join(
                        hr_directory_altitude,
                        f"{drone_id}_{mission_id}_{altitude}_{burst_number}.png",
                    )
                    print(f"Copying {filename} to {dest_path}")
                    shutil.copy(os.path.join(root, filename), dest_path)


if __name__ == "__main__":
    main()
