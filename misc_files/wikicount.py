import argparse
import os

def count_lines_in_directory(directory_path: str) -> int:
    total_lines = 0

    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(".json"):
                with open(os.path.join(dirpath, filename), 'r') as f:
                    lines = f.readlines()
                total_lines += len(lines)
                
                # print a message each time an additional million lines are counted
                if total_lines % 1000000 == 0:
                    print(f"Counted {total_lines} lines so far.")
    
    return total_lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count the number of lines in all JSON files in a directory.')
    parser.add_argument('directory', type=str, help='The path to the directory.')
    args = parser.parse_args()

    print("Starting the line count...")
    line_count = count_lines_in_directory(args.directory)
    print(f"Done! The JSON files in {args.directory} and its subdirectories have a total of {line_count} lines.")
