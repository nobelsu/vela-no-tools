import json

def read_text(path: str, encoding: str = "utf-8") -> str:
    with open(path, mode="r", encoding=encoding) as f:
        return f.read()
    
def write_text(path: str, content: str, encoding: str = "utf-8") -> None:
    with open(path, mode="w", encoding=encoding, newline="\n") as f:
        f.write(content)

def row_to_formatted_string(row: dict) -> str:
    """
    Convert a CSV row (dict) into the required formatted string
    and pass it to helper function f().
    """
    
    # Helper that safely loads JSON fields from CSV
    def parse_json_field(field):
        if not field or field.strip() == "":
            return []
        try:
            return json.loads(field)
        except Exception:
            return []   # fallback
    
    industry = row.get("industry", "")
    ipos = parse_json_field(row.get("ipos"))
    acquisitions = parse_json_field(row.get("acquisitions"))
    educations_json = parse_json_field(row.get("educations_json"))
    jobs_json = parse_json_field(row.get("jobs_json"))
    anonymised_prose = row.get("anonymised_prose", "").strip()

    # Build formatted multiline string
    formatted = f"""
        industry: "{industry}",
        ipos: {json.dumps(ipos)},
        acquisitions: {json.dumps(acquisitions)},
        educations_json: {json.dumps(educations_json)},
        jobs_json: {json.dumps(jobs_json)},
        anonymised_prose: \"\"\"
            {anonymised_prose}
        \"\"\"
        """

    # Pass to helper function f
    return formatted