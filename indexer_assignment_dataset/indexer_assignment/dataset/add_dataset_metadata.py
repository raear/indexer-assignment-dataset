from .config import CONFIG as cfg
import gzip
import json
import pandas as pd


def load_dataset(dataset_path):
    dataset = {}
    for line in open(dataset_path):
        line = line.strip()
        data = line.split("|")
        assert len(data) == 2
        pmid = int(data[0])
        assert 0 < pmid < 40000000
        indexer_id = int(data[1])
        assert 0 < indexer_id < 200
        dataset[pmid] = indexer_id
    return dataset


def main():

    contractor_dataset_path = cfg["contractor_dataset_path"]
    in_house_dataset_path = cfg["in_house_dataset_path"]
    contractor_dataset_with_metadata_path = cfg["contractor_dataset_with_metadata_path"]
    in_house_dataset_with_metadata_path = cfg["in_house_dataset_with_metadata_path"]

    start_file_num = 1
    end_file_num = cfg["medline_data_num_files"]
    extracted_data_path_template = cfg["extracted_data_path_template"]

    contractor_dataset = load_dataset(contractor_dataset_path)
    in_house_dataset   = load_dataset(in_house_dataset_path)

    with gzip.open(contractor_dataset_with_metadata_path, mode="wt", encoding="utf8") as contractor_dataset_write_file, \
         gzip.open(in_house_dataset_with_metadata_path,   mode="wt", encoding="utf8") as in_house_dataset_write_file:

        contractor_first = True
        in_house_first = True

        start = "["
        line_end = ",\n"
        end = "]"

        contractor_dataset_write_file.write(start)
        in_house_dataset_write_file.write(start)
        for file_num in range(start_file_num, end_file_num + 1):
            print(f"Adding metadata: {file_num:04}/{end_file_num:04}", end="\r")

            extracted_data_path = extracted_data_path_template.format(file_num)
            with gzip.open(extracted_data_path, "rt", encoding="utf8") as read_file:
                extracted_data = json.load(read_file)
                for medline_data in extracted_data:
                    pmid = int(medline_data["pmid"])
                    
                    if pmid in contractor_dataset:
                        medline_data["indexer_id"] = contractor_dataset[pmid]
                        if contractor_first:
                            contractor_first = False
                        else:
                            contractor_dataset_write_file.write(line_end)
                        example_str = json.dumps(medline_data, ensure_ascii=False)
                        contractor_dataset_write_file.write(example_str)

                    if pmid in in_house_dataset:
                        medline_data["indexer_id"] = in_house_dataset[pmid]
                        if in_house_first:
                            in_house_first = False
                        else:
                            in_house_dataset_write_file.write(line_end)
                        example_str = json.dumps(medline_data, ensure_ascii=False)
                        in_house_dataset_write_file.write(example_str)

        contractor_dataset_write_file.write(end)
        in_house_dataset_write_file.write(end)
                
    print(" "*50, end="\r")
   

if __name__ == "__main__":
    main()