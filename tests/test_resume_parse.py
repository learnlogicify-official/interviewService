from app.resume_parse import parse_resume

FRESHER = """Sachin K
sachin@example.com
SUMMARY
Final-year CSE student who likes backend systems.
PROJECTS
Learning Management System - Java, Spring Boot
Built an online judge that compiles Java submissions and runs test cases.
Reduced judge time by compiling once and reusing the class for every test case.
EDUCATION
Kalaignarkarunanidhi
Institute of Technology, Coimbatore
B.E. Computer Science and Engineering
CGPA 8.1 2021 - 2025
SKILLS
Java, Spring Boot, MySQL, Git
"""


def test_fresher_college_name_is_joined():
    d = parse_resume(FRESHER)
    names = [e["name"] for e in d["education"]]
    assert names == ["Kalaignarkarunanidhi Institute of Technology, Coimbatore"]
    assert "B.E." in d["education"][0]["degree"]


def test_fresher_questions_never_mention_work():
    d = parse_resume(FRESHER)
    assert d["career_stage"] == "fresher"
    plan = d["question_plan"]
    assert plan[0]["anchor"].startswith("Learning Management System")
    for item in plan:
        assert "your work" not in item["question"].lower()
        assert "on the job" not in item["question"].lower()


def test_real_job_is_experienced():
    d = parse_resume(
        "EXPERIENCE\nSoftware Engineer at Acme Corp\nJan 2022 - Present\n"
        "Built payment APIs used by 2M users.\nSKILLS\nGo, Postgres\n"
    )
    assert d["career_stage"] == "experienced"
    assert d["question_plan"][0]["anchor"] == "Software Engineer at Acme Corp"
