"""Split a resume by its section headers, then fill projects / jobs / skills separately."""

from __future__ import annotations

import re
from typing import Any

# Canonical section -> header phrases (matched on a short header line, not inside a sentence).
_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": (
        "summary", "professional summary", "profile", "objective", "career objective",
        "about", "about me",
    ),
    "experience": (
        "experience", "work experience", "professional experience", "employment",
        "work history", "career history", "professional history",
    ),
    "internships": ("internship", "internships", "intern experience"),
    "projects": (
        "projects", "project", "personal projects", "academic projects",
        "key projects", "technical projects", "selected projects",
    ),
    "education": (
        "education", "academics", "academic background", "academic qualifications",
        "qualifications",
    ),
    "skills": (
        "skills", "technical skills", "tech skills", "technologies", "tools",
        "core competencies", "competencies", "technical competencies",
    ),
    "certifications": (
        "certifications", "certification", "certificates", "certificate",
        "licenses", "licences",
    ),
    "achievements": ("achievements", "awards", "honors", "honours"),
}

_HEADER_TO_KEY = {}
for _key, _aliases in _HEADER_ALIASES.items():
    for _alias in _aliases:
        _HEADER_TO_KEY[_alias] = _key


def _clean_line(ln: str) -> str:
    s = (ln or "").replace("\u2022", " ").replace("●", " ").replace("▪", " ")
    s = re.sub(r"^[\-\*\u00b7]+\s*", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def _header_key(ln: str) -> str | None:
    """Return a section key when the line is a resume header, not a sentence."""
    raw = (ln or "").strip()
    if not raw or len(raw) > 42:
        return None
    if raw.count(" ") > 5:
        return None
    # A sentence, not a heading.
    if re.search(r"[.!?]", raw) and len(raw) > 18:
        return None
    label = raw.strip(" :|-–—").lower()
    label = re.sub(r"\s+", " ", label)
    return _HEADER_TO_KEY.get(label)


def is_header_line(ln: str) -> bool:
    return _header_key(ln) is not None


def split_sections(resume_text: str) -> tuple[dict[str, list[str]], list[str]]:
    """
    Group lines under the nearest header.

    Lines before the first header stay in "preamble".
    Returns (sections, header keys in order).
    """
    sections: dict[str, list[str]] = {"preamble": []}
    found: list[str] = []
    current = "preamble"
    for raw in (resume_text or "").splitlines():
        if not raw.strip():
            if current in sections and sections[current] and sections[current][-1] != "":
                sections[current].append("")
            continue
        key = _header_key(raw)
        if key:
            current = key
            if key not in sections:
                sections[key] = []
            if key not in found:
                found.append(key)
            continue
        line = _clean_line(raw)
        if not line:
            continue
        sections.setdefault(current, []).append(line)
    for key, lines in sections.items():
        sections[key] = [ln for ln in lines if ln]
    return sections, found


def _looks_like_new_entry(ln: str) -> bool:
    """A role, school, or project title — not a short bullet that starts with a verb."""
    if re.match(
        r"(?i)^(built|developed|designed|implemented|worked|led|created|used|helped|"
        r"responsible|managed|wrote|improved|reduced|increased|collaborated|handled)\b",
        ln,
    ):
        return False
    if re.search(r"\b(19|20)\d{2}\b", ln) and len(ln.split()) <= 14:
        return True
    if re.search(r"\s(?:at|@)\s", ln, re.I) and len(ln.split()) <= 14:
        return True
    return len(ln.split()) <= 8


def _chunks(lines: list[str], kind: str) -> list[list[str]]:
    """Group a section into entries. Verb bullets stay on the entry above them."""
    entries: list[list[str]] = []
    current: list[str] = []
    for ln in lines:
        starts = False
        if current:
            if kind in {"experience", "internships"}:
                starts = bool(re.search(r"\s(?:at|@)\s", ln, re.I)) and _looks_like_new_entry(ln)
            elif kind == "projects":
                has_body = any(
                    re.match(
                        r"(?i)^(built|developed|designed|implemented|worked|led|created|used|helped)\b",
                        prev,
                    )
                    or len(prev.split()) > 10
                    for prev in current[1:]
                )
                starts = has_body and _looks_like_new_entry(ln)
            else:
                starts = _looks_like_new_entry(ln)
        if starts:
            entries.append(current)
            current = [ln]
        else:
            current.append(ln)
    if current:
        entries.append(current)
    return entries


def _split_tokens(text: str) -> list[str]:
    parts = re.split(r"[,|/•·]|(?:\s+-\s+)|(?:\s+•\s+)", text)
    out: list[str] = []
    for part in parts:
        name = re.sub(r"\s+", " ", part).strip(" .;:-")
        if 2 <= len(name) <= 40 and not _header_key(name):
            out.append(name)
    return out


def _role_company(line: str) -> tuple[str, str]:
    s = _clean_line(line)
    at = re.split(r"\s+(?:at|@)\s+", s, maxsplit=1, flags=re.I)
    if len(at) == 2 and at[0].strip() and at[1].strip():
        return at[1].strip(" |-–—")[:80], at[0].strip(" |-–—")[:80]
    bits = re.split(r"\s+[\-–—|]\s+", s, maxsplit=1)
    if len(bits) == 2 and 1 <= len(bits[0].split()) <= 8:
        return bits[0].strip()[:80], bits[1].strip()[:80]
    return s[:80], ""


def _project_name(line: str) -> str:
    s = _clean_line(line)
    headed = re.split(r"\s+[\-–—:|]\s+", s, maxsplit=1)[0].strip()
    words = headed.split()
    if 1 <= len(words) <= 8 and len(headed) <= 70:
        return headed
    return " ".join(words[:6])[:70] or "project"


def _entry(kind: str, lines: list[str]) -> dict[str, Any] | None:
    body = [ln for ln in lines if ln]
    if not body:
        return None
    head, rest = body[0], body[1:]
    claim = " ".join(rest)[:280]
    if kind == "projects":
        name = _project_name(head)
        if _header_key(name):
            return None
        return {"name": name[:90], "stack": [], "claim": (claim or head)[:280]}
    if kind in {"experience", "internships"}:
        company, role = _role_company(head)
        if _header_key(company):
            return None
        return {
            "company": company[:80],
            "role": role[:80],
            "claim": (claim or head)[:240],
        }
    return None


def _questions(dossier: dict[str, Any]) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []

    def add(anchor: str, question: str) -> None:
        anchor = anchor.strip()
        question = " ".join(question.split()).strip()
        if anchor and question:
            plan.append({"anchor": anchor[:120], "question": question[:400]})

    fresher = (dossier.get("career_stage") or career_stage(dossier)) == "fresher"

    def ask_jobs() -> None:
        for job in dossier.get("internships") or []:
            company = str(job.get("company") or "").strip()
            role = str(job.get("role") or "").strip()
            who = f"{role} at {company}" if role and company else (company or role)
            if not who:
                continue
            if str(job.get("kind") or "") == "internship":
                add(
                    who,
                    f"During your internship as {who}, what was one task you owned end to end, "
                    "and what did you learn when it did not work the first time?",
                )
            else:
                add(
                    who,
                    f"You listed {who} on your resume. What did you personally own there, "
                    "and what broke or nearly broke?",
                )

    def ask_projects() -> None:
        for project in dossier.get("projects") or []:
            name = str(project.get("name") or "").strip()
            if not name:
                continue
            if fresher:
                add(
                    name,
                    f"Tell me about your {name} project — what problem it solved, which part you "
                    "built yourself, and the hardest bug you hit.",
                )
            else:
                add(
                    name,
                    f"On your {name} project, what did you personally own versus the team, "
                    "and what broke when it was used for real?",
                )

    def ask_certs() -> None:
        for cert in dossier.get("certifications") or []:
            name = str(cert.get("name") or "").strip()
            if name:
                add(
                    name,
                    f"Your resume lists {name}. What is one thing from it you have actually built "
                    "or practised, and where did it fall short?",
                )

    def ask_skills() -> None:
        for skill in (dossier.get("skills") or [])[:4]:
            name = str(skill.get("name") or skill or "").strip()
            if len(name) >= 2:
                where = "in a project or assignment" if fresher else "on real work"
                add(
                    name,
                    f"Your resume lists {name}. Give one concrete place you used it {where}, "
                    "and one trade-off you made.",
                )

    def ask_education() -> None:
        for edu in dossier.get("education") or []:
            name = str(edu.get("name") or "").strip()
            if not name:
                continue
            degree = str(edu.get("degree") or "").strip()
            course = f" ({degree})" if degree and degree != name else ""
            if fresher:
                add(
                    name,
                    f"At {name}{course}, which course or college project taught you the most, "
                    "and what did you build with it?",
                )
            else:
                add(
                    name,
                    f"From your studies at {name}{course}, which concept do you still rely on "
                    "most in your current work, and why?",
                )

    # Freshers: projects are the strongest evidence; experienced: lead with jobs.
    order = (
        (ask_projects, ask_jobs, ask_skills, ask_certs, ask_education)
        if fresher
        else (ask_jobs, ask_projects, ask_certs, ask_skills, ask_education)
    )
    for step in order:
        step()
    if not plan:
        if fresher:
            add(
                "resume",
                "Walk me through the project on your resume you are most proud of — what it does, "
                "what you built yourself, and the hardest bug you fixed.",
            )
        else:
            add(
                "resume",
                "Walk me through the most technically demanding work on your resume — "
                "your role, the hardest bug, and how you measured success.",
            )
    return plan[:12]


def parse_resume(resume_text: str) -> dict[str, Any]:
    """Section-aware dossier. Work history is not stored as a project."""
    sections, found = split_sections(resume_text or "")
    projects: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    skills: list[dict[str, str]] = []
    certifications: list[dict[str, str]] = []
    education: list[dict[str, str]] = []

    for key in ("projects",):
        for entry in _chunks(sections.get(key) or [], "projects"):
            row = _entry("projects", entry)
            if row and len(projects) < 6:
                projects.append(row)

    for key in ("experience", "internships"):
        for entry in _chunks(sections.get(key) or [], key):
            row = _entry(key, entry)
            if row and len(jobs) < 6:
                row["kind"] = _job_kind(key, row)
                jobs.append(row)

    skill_lines = sections.get("skills") or []
    for ln in skill_lines:
        for name in _split_tokens(ln):
            if len(skills) < 16:
                skills.append({"name": name, "evidence": "skills section"})

    for ln in sections.get("certifications") or []:
        name = _clean_line(ln)
        if 3 <= len(name) <= 80 and len(certifications) < 8:
            certifications.append({"name": name, "issuer": ""})

    for row in _education_entries(sections.get("education") or []):
        if len(education) < 4:
            education.append(row)

    preamble = sections.get("preamble") or []
    summary_lines = sections.get("summary") or preamble[:2]
    summary = " ".join(summary_lines)[:180] if summary_lines else "Resume on file."

    # No headers: do not call every line a project. Classify each line on its own.
    if not found:
        projects, jobs, skills, certifications, education, summary = _parse_without_headers(
            resume_text or ""
        )

    dossier: dict[str, Any] = {
        "summary": summary or "Resume on file.",
        "projects": projects,
        "internships": jobs,
        "certifications": certifications,
        "skills": skills,
        "education": education,
        "sections_found": found,
        "source": "sections" if found else "lines",
    }
    dossier["career_stage"] = career_stage(dossier)
    dossier["question_plan"] = _questions(dossier)
    return dossier


_SCHOOL_RE = re.compile(
    r"(?i)\b(university|college|institute|institution|school|academy|polytechnic|iit|nit|iiit)\b"
)
_DEGREE_RE = re.compile(
    r"(?i)\b(bachelor|master|b\.?\s?tech|b\.?\s?e\b|m\.?\s?tech|m\.?\s?e\b|b\.?\s?sc|m\.?\s?sc|"
    r"bca|mca|mba|ph\.?\s?d|diploma|hsc|sslc|higher secondary|secondary|class\s+(x|xii|10|12)|"
    r"12th|10th|degree)\b"
)
_EDU_DETAIL_RE = re.compile(r"(?i)\b(cgpa|gpa|percentage|%|grade|(19|20)\d{2})\b")


def _education_entries(lines: list[str]) -> list[dict[str, str]]:
    """
    One entry per school. A college name split across lines
    ("Kalaignarkarunanidhi" / "Institute of Technology, Coimbatore") is joined back.
    """
    entries: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None

    def new() -> dict[str, Any]:
        row = {"school": "", "degree": "", "detail": [], "school_done": False}
        entries.append(row)
        return row

    for ln in lines:
        s = _clean_line(ln)
        if not s:
            continue
        is_school = bool(_SCHOOL_RE.search(s))
        is_degree = bool(_DEGREE_RE.search(s))
        if is_school:
            if cur is None or cur["school_done"]:
                cur = new()
            cur["school"] = f"{cur['school']} {s}".strip() if cur["school"] else s
            cur["school_done"] = True
            if is_degree and not cur["degree"]:
                cur["degree"] = s
            continue
        if is_degree:
            if cur is None or cur["degree"]:
                cur = new()
            cur["degree"] = s
            continue
        if _EDU_DETAIL_RE.search(s) or len(s.split()) > 8:
            if cur is None:
                cur = new()
            cur["detail"].append(s)
            continue
        # Short line with no school keyword: start of a name that continues on the next line.
        if cur is None or cur["school_done"]:
            cur = new()
        cur["school"] = f"{cur['school']} {s}".strip()

    out: list[dict[str, str]] = []
    for row in entries:
        name = row["school"] or row["degree"]
        if not name or _header_key(name):
            continue
        detail = " · ".join(x for x in [row["degree"] if row["school"] else "", *row["detail"]] if x)
        out.append({"name": name[:140], "degree": row["degree"][:120], "detail": detail[:180]})
    return out


def _job_kind(section: str, row: dict[str, Any]) -> str:
    blob = f"{row.get('role') or ''} {row.get('company') or ''} {row.get('claim') or ''}"
    if section == "internships" or re.search(r"(?i)\bintern(ship)?\b", blob):
        return "internship"
    return "job"


def career_stage(dossier: dict[str, Any]) -> str:
    """'experienced' only with a real (non-intern) job; otherwise a student / fresher."""
    for job in dossier.get("internships") or []:
        kind = str(job.get("kind") or "") or _job_kind("experience", job)
        if kind == "job":
            return "experienced"
    return "fresher"


_EDU_RE = re.compile(
    r"(?i)\b(university|college|institute|bachelor|master|b\.?\s?tech|b\.?\s?e\b|"
    r"m\.?\s?tech|gpa|cgpa|high school|diploma)\b"
)
_CERT_RE = re.compile(r"(?i)\b(certified|certification|certificate)\b")
_JOB_RE = re.compile(
    r"(?i)\b(engineer|developer|intern|analyst|manager|designer)\b.+\b(at|@)\b|"
    r".+\s+[\-–—|]\s+.+\b(19|20)\d{2}\b"
)


def _parse_without_headers(text: str) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    str,
]:
    projects: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    skills: list[dict[str, str]] = []
    certifications: list[dict[str, str]] = []
    education: list[dict[str, str]] = []
    loose: list[str] = []
    lines = [_clean_line(ln) for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 3]
    edu_lines: list[str] = []
    for ln in lines:
        if _EDU_RE.search(ln):
            # "Kalaignarkarunanidhi" / "Institute of Technology, ..." — rejoin the split name.
            if (
                re.match(r"(?i)^(institute|college|university|school|academy)\b", ln)
                and loose
                and len(loose[-1].split()) <= 3
                and not edu_lines
            ):
                edu_lines.append(loose.pop())
            edu_lines.append(ln)
        elif _CERT_RE.search(ln):
            certifications.append({"name": ln[:80], "issuer": ""})
        elif ln.count(",") >= 3 and len(ln) < 200:
            for name in _split_tokens(ln):
                skills.append({"name": name, "evidence": "resume line"})
        elif _JOB_RE.search(ln):
            company, role = _role_company(ln)
            job = {"company": company, "role": role, "claim": ln[:240]}
            job["kind"] = _job_kind("experience", job)
            jobs.append(job)
        elif re.match(r"(?i)^project\b", ln):
            projects.append({
                "name": _project_name(ln),
                "stack": [],
                "claim": ln[:280],
            })
        else:
            loose.append(ln)
    education = _education_entries(edu_lines)
    summary = " ".join(loose[:2])[:180] if loose else "Resume on file."
    return projects[:6], jobs[:6], skills[:16], certifications[:8], education[:4], summary
