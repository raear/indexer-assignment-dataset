import datetime
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import pytrec_eval


LONG_MONTHS =   ["january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december"]


SHORT_MONTHS = ["jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec"]


def load_qrels(path):
    qrels = pytrec_eval.parse_qrel(open(path))
    qrels_mod = { int(q_id): [int(p_id) for p_id in p_id_list] for q_id, p_id_list in qrels.items()}
    return qrels_mod


def load_run(path):
    run = pytrec_eval.parse_run(open(path))
    run_mod = { int(q_id): { int(p_id): float(score) for p_id, score in p_score_dict.items() } for q_id, p_score_dict in run.items()}
    return run_mod


def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_citation_node(medline_citation_node):

    pmid_node = medline_citation_node.find("PMID")
    pmid = pmid_node.text.strip()
    pmid = int(pmid)

    title = ""
    title_node = medline_citation_node.find("Article/ArticleTitle") 
    title = ET.tostring(title_node, encoding="unicode", method="text")
    title = title.strip() if title is not None else ""

    vernacular_title = ""
    vernacular_title_node = medline_citation_node.find("Article/VernacularTitle")
    if vernacular_title_node is not None:
        vernacular_title = ET.tostring(vernacular_title_node, encoding="unicode", method="text")
        vernacular_title = vernacular_title.strip() if vernacular_title is not None else ""
    
    abstract = ""
    abstract_node = medline_citation_node.find("Article/Abstract")
    if abstract_node is not None:
        abstract_text_nodes = abstract_node.findall("AbstractText")
        for abstract_text_node in abstract_text_nodes:
            if "Label" in abstract_text_node.attrib:
                if len(abstract) > 0:
                    abstract += " "
                abstract += abstract_text_node.attrib["Label"].strip() + ": "
            abstract_text = ET.tostring(abstract_text_node, encoding="unicode", method="text")
            if abstract_text is not None:
                abstract += abstract_text.strip()

    date_completed_text = None
    date_completed_node = medline_citation_node.find("DateCompleted")
    if date_completed_node is not None:
        date_completed_year =  date_completed_node.find("Year").text.strip()
        date_completed_month = date_completed_node.find("Month").text.strip()
        date_completed_day =   date_completed_node.find("Day").text.strip()
        date_completed = datetime.date(int(date_completed_year), int(date_completed_month), int(date_completed_day))
        date_completed_text = date_completed.isoformat()

    journal_nlmid_node = medline_citation_node.find("MedlineJournalInfo/NlmUniqueID")
    journal_nlmid = journal_nlmid_node.text.strip() if journal_nlmid_node is not None else None

    medline_ta = None
    medline_ta_node = medline_citation_node.find("MedlineJournalInfo/MedlineTA")
    if medline_ta_node is not None and medline_ta_node.text is not None:
        medline_ta = medline_ta_node.text.strip()

    journal_volume = None
    journal_volume_node = medline_citation_node.find("Article/Journal/JournalIssue/Volume")
    if journal_volume_node is not None:
        journal_volume = journal_volume_node.text.strip()

    journal_issue = None
    journal_issue_node = medline_citation_node.find("Article/Journal/JournalIssue/Issue")
    if journal_issue_node is not None:
        journal_issue = journal_issue_node.text.strip()

    medline_date_text = None
    pub_date_text = None
    pub_date_node = medline_citation_node.find("Article/Journal/JournalIssue/PubDate")
    medlinedate_node = pub_date_node.find("MedlineDate")
    if medlinedate_node is not None:
        medline_date_text = medlinedate_node.text.strip().lower()
    else:
        pub_year = int(pub_date_node.find("Year").text.strip())
        pub_date_text = f"{pub_year:04d}"

        season_node = pub_date_node.find("Season")
        month_node =  pub_date_node.find("Month")
        day_node =    pub_date_node.find("Day")
      
        if season_node is not None:
            season = season_node.text.strip().lower()
            pub_date_text += f" {season}"
        elif month_node is not None:
            month = month_node.text.strip(" .").lower()
            month_num = None
            if month.isnumeric():
                month_num = int(month)
            elif month in SHORT_MONTHS:
                month_num = SHORT_MONTHS.index(month) + 1
            elif month in LONG_MONTHS:
                month_num = LONG_MONTHS.index(month) + 1

            if month_num:
                pub_date_text += f"-{month_num:02d}"
            else:
                pub_date_text += f"_{month}"

            if day_node is not None:
                day = int(day_node.text.strip())
                pub_date_text += f"-{day:02d}"

    medline_citation_node_attribs = medline_citation_node.attrib
    indexing_method_attrib_name = "IndexingMethod"
    indexing_method = medline_citation_node_attribs[indexing_method_attrib_name].strip().lower() if indexing_method_attrib_name in medline_citation_node_attribs else "human"

    citation_data = {"pmid": pmid, 
                     "medline_date": remove_ws(medline_date_text),
                     "pub_date": pub_date_text,
                     "volume": journal_volume,
                     "issue": journal_issue,
                     "indexing_method": indexing_method,
                     "date_completed": date_completed_text,
                     "title": remove_ws(title), 
                     "vernacular_title": remove_ws(vernacular_title), 
                     "abstract": remove_ws(abstract),
                     "medline_ta": medline_ta,
                     "journal_nlmid": journal_nlmid,}

    return citation_data


def remove_ws(text):
    if text is None:
        return text
    text = text.replace("\n", "").replace("\t", "")
    return text


def save_qrels(path, qrels, max_results=1000000000):
    # q_id	Q0	p_id	1
    data = { "q_id": [], "Q0": [], "p_id": [], "1": [], }
    for q_id in sorted(qrels):
        for p_id in sorted(qrels[q_id])[:max_results]:
            data["q_id"].append(str(q_id))
            data["Q0"].append("Q0")
            data["p_id"].append(str(p_id))
            data["1"].append("1")
    df = pd.DataFrame(data, columns=["q_id", "Q0", "p_id", "1"])
    df.to_csv(path, index=False, header=False, sep="\t")


def save_run(path, qrels, run_id="runid1", max_results=1000000000):
    # 1 Q0 pid1    1 2.73 runid1
    data = { "q_id": [], "Q0": [], "p_id": [], "rank": [], "score": [], "run_id": [] }
    for q_id in qrels:
        for p_idx, p_item in enumerate(sorted(qrels[q_id].items(), key=lambda x: x[1], reverse=True)[:max_results]):
            p_id, score = p_item
            rank = p_idx + 1
            data["q_id"].append(str(q_id))
            data["Q0"].append("Q0")
            data["p_id"].append(str(p_id))
            data["rank"].append(str(rank))
            data["score"].append(f"{score:.20f}")
            data["run_id"].append(run_id)
    df = pd.DataFrame(data, columns=["q_id", "Q0", "p_id", "rank", "score", "run_id"])
    df.to_csv(path, index=False, header=False, sep="\t")


def to_date(date_time_str):
    if not date_time_str:
        return None

    date_time_obj = datetime.datetime.strptime(date_time_str, "%Y-%m-%d")
    date = date_time_obj.date()
    return date