def merge_chunk(split_files, output_file):
    with open(output_file, 'wb') as outfile:
        for split_file in split_files:
            with open(split_file, 'rb') as infile:
                outfile.write(infile.read())
