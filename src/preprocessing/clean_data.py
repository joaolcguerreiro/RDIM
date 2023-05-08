import shutil
import os
import csv


def main():
    # Open CSV metadata file
    try:
        with open(os.path.join('..', '..', 'data', 'manifest-160070915466', 'metadata.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)

            # Skip first row (Column names)
            next(reader)

            # Desired index
            modality_index = 10
            path_index = 15

            # Parse path function
            parse_path = lambda path: os.path.join('..', '..', 'data', 'manifest-1600709154662', os.path.dirname(path)[2:])

            # Clean data
            for row in reader:
                if row[modality_index] != 'CT':
                    shutil.rmtree(parse_path(row[path_index]))
    except Exception as e:
        print("Download data again. An error occurred: {}".format(e))


if __name__ == '__main__':
    main()
