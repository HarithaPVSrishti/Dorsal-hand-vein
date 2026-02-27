import os
import shutil
import re

def combine_datasets():
    # Updated paths for organized folder structure
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "Data")
    
    db1_path = os.path.join(data_path, "DorsalHandVeins_DB1_png")
    db2_path = os.path.join(data_path, "DorsalHandVeins_DB2_png")
    output_path = os.path.join(data_path, "Total_Vein_Dataset")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")

    # Helper to extract person number from filename
    # Example: person_054_db1_L1.png -> 54
    def get_person_num(filename):
        match = re.search(r'person_(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    # Processing DB1
    print("Processing Database 1...")
    db1_files = [f for f in os.listdir(db1_path) if f.endswith('.png')]
    db1_max_person = 0
    for f in db1_files:
        p_num = get_person_num(f)
        if p_num is not None:
            new_p_name = f"person_{p_num:03d}"
            target_dir = os.path.join(output_path, new_p_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            target_file = os.path.join(target_dir, f)
            if not os.path.exists(target_file):
                shutil.copy2(os.path.join(db1_path, f), target_file)
            
            if p_num > db1_max_person:
                db1_max_person = p_num

    print(f"DB1 check/copy completed. Max person ID: {db1_max_person}.")

    # Processing DB2
    print("Processing Database 2...")
    offset = db1_max_person
    db2_files = [f for f in os.listdir(db2_path) if f.endswith('.png')]
    db2_max_person = 0
    
    for f in db2_files:
        p_num = get_person_num(f)
        if p_num is not None:
            new_p_num = p_num + offset
            new_p_name = f"person_{new_p_num:03d}"
            target_dir = os.path.join(output_path, new_p_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            new_filename = f.replace(f"person_{p_num:03d}", f"person_{new_p_num:03d}")
            target_file = os.path.join(target_dir, new_filename)
            
            if not os.path.exists(target_file):
                shutil.copy2(os.path.join(db2_path, f), target_file)
            
            if p_num > db2_max_person:
                db2_max_person = p_num

    print(f"DB2 completed. Added {db2_max_person} people starting from {offset + 1}.")
    print(f"Total people in combined dataset: {offset + db2_max_person}")
    print(f"Dataset path: {output_path}")

if __name__ == "__main__":
    combine_datasets()
