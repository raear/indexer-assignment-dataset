from .config import CONFIG as cfg
import pickle
import xml.etree.ElementTree as ET


def main():    
    lookup = {}
    root_node = ET.parse(cfg["lsi_path"])
    for serial_node in root_node.findall("Serial"):
        nlm_unique_id_node = serial_node.find("NlmUniqueID")
        nlmid = nlm_unique_id_node.text.strip()
        journal_descriptor_set = set()
        broad_journal_heading_list_node = serial_node.find("BroadJournalHeadingList")
        if broad_journal_heading_list_node is not None:
            for broad_journal_heading_node in broad_journal_heading_list_node.findall("BroadJournalHeading"):
                broad_journal_heading = broad_journal_heading_node.text.strip()
                journal_descriptor_set.add(broad_journal_heading)
        if journal_descriptor_set:
            lookup[nlmid] = journal_descriptor_set

    pickle.dump(lookup, open(cfg["journal_descriptor_lookup_path"], "wb"))


if __name__ == "__main__":
    main()