"""Step 0 - CHB-MIT header audit.

Reads the EDF+ header of every recording (no signal data loaded) and produces
output/00_channel_audit.csv and output/00_channel_audit_summary.md.

Header is parsed by hand rather than via mne: mne.io.read_raw_edf renames
duplicate channel labels (e.g. the two "T8-P8" channels become "T8-P8-0" /
"T8-P8-1"), which would hide the exact duplicate-derivation issue this audit
needs to detect. See CLAUDE.md section 4.3.
"""

import time
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "CHB-MIT Dataset"
OUTPUT_DIR = BASE_DIR / "output"

# The 21 unique derivations after dropping P7-T7 (polarity-flipped copy of
# T7-P7) and the second T8-P8 occurrence. See CLAUDE.md section 4.3.
REQUIRED_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
    "T7-FT9", "FT9-FT10", "FT10-T8",
]

# Channels dropped at load time in preprocessing (not "foreign" - real
# derivations that duplicate information already in REQUIRED_CHANNELS).
KNOWN_DUPLICATE_CHANNELS = {"P7-T7"}

GLASS7_DEPENDENCY = ["T7-FT9", "FT9-FT10", "FT10-T8"]


def read_edf_header(path):
    """Parse the EDF+ fixed-format ASCII header. Reads only the header block."""
    with open(path, "rb") as f:
        f.read(8 + 80 + 80 + 8 + 8)  # version, patient id, recording id, start date, start time
        f.read(8)   # n_header_bytes (derivable from ns; not needed)
        f.read(44)  # reserved
        n_records = int(f.read(8))
        record_dur = float(f.read(8))
        ns = int(f.read(4))
        labels = [f.read(16).decode("ascii", errors="replace").strip() for _ in range(ns)]
        f.read(80 * ns)  # transducer type
        f.read(8 * ns)   # physical dimension
        f.read(8 * ns)   # physical minimum
        f.read(8 * ns)   # physical maximum
        f.read(8 * ns)   # digital minimum
        f.read(8 * ns)   # digital maximum
        f.read(80 * ns)  # prefiltering
        n_samples = [int(f.read(8)) for _ in range(ns)]

    duration_sec = n_records * record_dur
    sampling_rates = [n / record_dur if record_dur > 0 else float("nan") for n in n_samples]
    return {"labels": labels, "duration_sec": duration_sec, "sampling_rates": sampling_rates}


def parse_summary_seizure_counts(summary_path):
    """filename -> n_seizures, read from 'File Name:' / 'Number of Seizures in File:' lines."""
    counts = {}
    current_file = None
    with open(summary_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name"):
                current_file = line.split(":", 1)[-1].strip()
            elif line.startswith("Number of Seizures in File") and current_file is not None:
                counts[current_file] = int(line.split(":")[-1].strip())
    return counts


def is_monopolar(labels):
    """True if none of the 21 required bipolar derivations are present by exact name.

    Determined from the header data itself (not a hardcoded filename list) so
    the audit still works if the dataset changes.
    """
    return not any(ch in labels for ch in REQUIRED_CHANNELS)


def dominant_sampling_rate(sampling_rates):
    counts = defaultdict(int)
    for r in sampling_rates:
        counts[round(r, 3)] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def build_rows():
    subject_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("chb"))

    rows = []
    for subj_dir in subject_dirs:
        subject = subj_dir.name
        summary_path = subj_dir / f"{subject}-summary.txt"
        seizure_counts = parse_summary_seizure_counts(summary_path) if summary_path.exists() else {}

        for edf_path in sorted(subj_dir.glob("*.edf")):
            header = read_edf_header(edf_path)
            labels = header["labels"]
            rate = dominant_sampling_rate(header["sampling_rates"])

            row = {
                "subject": subject,
                "filename": edf_path.name,
                "duration_sec": header["duration_sec"],
                "sampling_rate": rate,
                "n_channels_total": len(labels),
                "channel_names": ";".join(labels),
                "n_seizures": seizure_counts.get(edf_path.name, 0),
                "is_monopolar": is_monopolar(labels),
            }
            for ch in REQUIRED_CHANNELS:
                row[f"has_{ch}"] = ch in labels
            rows.append(row)
    return rows


def write_csv(rows, path):
    import pandas as pd

    columns = (
        ["subject", "filename", "duration_sec", "sampling_rate", "n_channels_total", "channel_names"]
        + [f"has_{ch}" for ch in REQUIRED_CHANNELS]
        + ["n_seizures", "is_monopolar"]
    )
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)
    return df


def write_summary(df, path):
    total_duration_sec = df["duration_sec"].sum()
    total_hours = total_duration_sec / 3600
    total_seizures = int(df["n_seizures"].sum())
    n_subjects = df["subject"].nunique()
    n_files = len(df)

    lines = []
    lines.append("# CHB-MIT Header Audit Summary")
    lines.append("")
    lines.append(f"- Subjects (raw folders, before chb01/chb21 merge): {n_subjects}")
    lines.append(f"- Files: {n_files}")
    lines.append(f"- Total recorded duration: {total_hours:.2f} hours ({total_duration_sec:.0f} s)")
    lines.append(f"- Total seizures (sum of 'Number of Seizures in File' across all files): {total_seizures}")
    lines.append("")

    # --- 1. Per-channel presence ---
    lines.append("## 1. Channel presence")
    lines.append("")
    lines.append("| Channel | Subjects (of %d) | Files (of %d) | %% of total duration |" % (n_subjects, n_files))
    lines.append("|---|---|---|---|")
    for ch in REQUIRED_CHANNELS:
        col = f"has_{ch}"
        subj_count = df.loc[df[col], "subject"].nunique()
        file_count = int(df[col].sum())
        dur_pct = 100 * df.loc[df[col], "duration_sec"].sum() / total_duration_sec
        lines.append(f"| {ch} | {subj_count} | {file_count} | {dur_pct:.1f}% |")
    lines.append("")

    # --- 2. Subjects missing Glass-7 dependency channels ---
    lines.append("## 2. Subjects missing T7-FT9, FT9-FT10, or FT10-T8 (Glass-7 dependency)")
    lines.append("")
    lines.append("**This determines whether Glass-7 survives as a configuration.**")
    lines.append("")
    for ch in GLASS7_DEPENDENCY:
        col = f"has_{ch}"
        present_subjects = set(df.loc[df[col], "subject"].unique())
        missing_subjects = sorted(set(df["subject"].unique()) - present_subjects)
        lines.append(f"- Missing `{ch}` (in at least one file): {missing_subjects if missing_subjects else 'none'}")
    # subjects missing the channel in ALL of their files (fully unusable for Glass-7)
    fully_missing = []
    for subject, g in df.groupby("subject"):
        if any(not g[f"has_{ch}"].any() for ch in GLASS7_DEPENDENCY):
            fully_missing.append(subject)
    lines.append("")
    lines.append(f"- Subjects missing at least one of these channels in **every** file "
                 f"(cannot support Glass-7 at all): {sorted(fully_missing) if fully_missing else 'none'}")
    lines.append("")

    # --- 3. Subjects with all 21 channels in every file ---
    lines.append("## 3. Subjects with all 21 required channels present in every file")
    lines.append("")
    complete_subjects = []
    for subject, g in df.groupby("subject"):
        if all(g[f"has_{ch}"].all() for ch in REQUIRED_CHANNELS):
            complete_subjects.append(subject)
    lines.append(f"- Count: {len(complete_subjects)} of {n_subjects}")
    lines.append(f"- Subjects: {sorted(complete_subjects)}")
    lines.append("")

    # --- 4. Foreign / non-standard channels per subject ---
    lines.append("## 4. Non-standard channels found (not in the 21 required + known duplicate P7-T7)")
    lines.append("")
    known = set(REQUIRED_CHANNELS) | KNOWN_DUPLICATE_CHANNELS
    foreign_by_subject = defaultdict(set)
    for _, row in df.iterrows():
        labels = row["channel_names"].split(";")
        for lab in labels:
            if lab not in known:
                foreign_by_subject[row["subject"]].add(lab)
    if foreign_by_subject:
        for subject in sorted(foreign_by_subject):
            labs = sorted(foreign_by_subject[subject])
            lines.append(f"- `{subject}`: {labs}")
    else:
        lines.append("- None found.")
    lines.append("")

    # --- 5. Sampling rate anomalies ---
    lines.append("## 5. Files where sampling rate != 256 Hz")
    lines.append("")
    bad_rate = df[df["sampling_rate"] != 256]
    if len(bad_rate):
        for _, row in bad_rate.iterrows():
            lines.append(f"- `{row['subject']}/{row['filename']}`: {row['sampling_rate']} Hz")
    else:
        lines.append("- None. All files are 256 Hz.")
    lines.append("")

    # --- 6. Totals + monopolar chb12 impact ---
    lines.append("## 6. Recording totals and chb12 monopolar exclusion")
    lines.append("")
    lines.append(f"- Total duration: {total_hours:.2f} hours")
    lines.append(f"- Total seizures (all files, all subjects, before any exclusion): {total_seizures}")
    monopolar = df[df["is_monopolar"]]
    monopolar_seizures = int(monopolar["n_seizures"].sum())
    lines.append(
        f"- Files auto-detected as monopolar (none of the 21 required bipolar derivations present): "
        f"{len(monopolar)} -> {sorted(monopolar['filename'].tolist())}"
    )
    lines.append(
        f"- Seizures in those monopolar files (removed if excluded per CLAUDE.md section 4.2): {monopolar_seizures}"
    )
    lines.append(
        f"- Total seizures after excluding monopolar files: {total_seizures - monopolar_seizures}"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `subject` here is the raw CHB-MIT folder id. chb01 and chb21 are the same patient "
        "(CLAUDE.md section 4.1) and will be merged into `chb01+21` starting from the preprocessing step, "
        "not in this audit."
    )
    lines.append(
        "- `is_monopolar` is derived purely from header content (absence of all 21 required bipolar "
        "channel names), not from a hardcoded filename list."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    rows = build_rows()
    df = write_csv(rows, OUTPUT_DIR / "00_channel_audit.csv")
    write_summary(df, OUTPUT_DIR / "00_channel_audit_summary.md")

    elapsed = time.time() - t0
    print(f"Audited {len(df)} files across {df['subject'].nunique()} subjects in {elapsed:.1f}s")
    print(f"-> {OUTPUT_DIR / '00_channel_audit.csv'}")
    print(f"-> {OUTPUT_DIR / '00_channel_audit_summary.md'}")


if __name__ == "__main__":
    main()
