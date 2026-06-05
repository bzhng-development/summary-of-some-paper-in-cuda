"""Two-pass prompts for The Peterman Post podcast-transcript summaries.

Design:
- P1 drafts the summary in the user's preferred style: chapter-by-chapter
  career/project sections, stories and lessons interwoven, inline direct
  quotes, bolded key-concept names, specific names/numbers/years preserved,
  light editorial voice.
- P2 receives the transcript + P1 draft and produces an IMPROVED FULL
  summary — fills gaps, tightens wording, adds the capstone paragraph.
  P2's output is the final deliverable (it SUBSUMES P1, does not append).
  This is the test-time-compute scaling pass.

The example summary (example_adrien.md) is attached for STYLE only. Strong
guardrails warn against repeating its content, names, or framing.
"""

from pathlib import Path


_EXAMPLE_PATH = Path(__file__).parent / "example_adrien.md"


def _example_block() -> str:
    """Load the Adrien example, wrap it with style-only guardrails."""
    body = _EXAMPLE_PATH.read_text(encoding="utf-8")
    return f"""<style_example>
Below is an example summary of a DIFFERENT episode (Adrien Friggeri). It is
provided ONLY to show structure, tone, density, and the way stories and
lessons are interwoven. DO NOT:
- mention Adrien Friggeri, Bento, Cameron Marlow, Mike Krieger, Instagram
  A/B testing, Clubhouse, Ray-Ban Stories, Sung Yu, Vincent Hardy, James
  Pierce, Alex Schultz, the "two-step adoption playbook," the "stealing
  scope" frame, or any other specific person/project/concept from this
  example unless THEY ALSO APPEAR in the input transcript you are summarising.
- copy the bolded subheadings verbatim. Make your own headings that fit the
  ACTUAL career arc / projects discussed in the input transcript.
- import the "rising tide" / "surface area for luck" / "Heinlein quote" /
  "Bitcoin in 2013" framings. Those belong to Adrien only.

What to copy:
- The chapter-by-chapter organisation (often by job/team/promo/project).
- Stories and lessons INTERWOVEN within each section — not split apart.
- Short inline direct quotes, in double-quotes, sprinkled for color.
- **Bold inline** for the key concept or named framework the guest coined.
- Specific names, numbers, years, system names preserved everywhere.
- Light editorial voice — parenthetical asides, naming a pattern the guest
  used ("his X playbook is...").
- A closing capstone paragraph that names cross-cutting themes — without
  referring to other guests unless the input transcript itself does.

--- BEGIN EXAMPLE (Adrien Friggeri — NOT the subject of your task) ---

{body}

--- END EXAMPLE ---
</style_example>"""


P1_SYSTEM = f"""You are writing a long-form, narrative summary of a podcast interview transcript from The Peterman Post (Ryan Peterman's podcast on technical careers). Your job is to capture every substantive thing the guest said, organized by the actual arc of their career / the projects discussed, with stories and lessons interwoven in the same prose.

{_example_block()}

Hard rules for your output:

1. **Structure.** Organize by career chapters / projects / phases as they appear in the transcript. Use Markdown bold headings like `**Heading here**` (NOT `##` headers — the example uses bold paragraphs). Each section should be 1-4 paragraphs of dense prose.

2. **Interweave story + lesson.** Do not separate "stories" from "learnings." Within each chapter, narrate what happened AND extract the principle or technique it illustrates, in the same flow.

3. **Quotes inline.** When the guest says something striking or load-bearing, quote it in double-quotes inside your prose. Short quotes (< 30 words) inline; longer quotes set off if needed.

4. **Bold the named concept.** When the guest coins a phrase, framework, or principle (e.g. their "X playbook," their "Y model"), bold it inline. When you name a pattern they used that they didn't name explicitly, bold that too.

5. **Preserve specifics.** Names of people, companies, systems, products, languages, frameworks. Years and dates. Numbers (team size, latency, dollars, percentages, lines of code). If they describe an architecture or a decision, capture the actual decision — not a vague summary.

6. **Editorial voice — light, parenthetical, never moralising.** Use parentheticals for asides, light interpretation, naming a pattern. Don't say "the takeaway is" — let the structure show it.

7. **Length.** Match what the transcript justifies. A 2-hour deep podcast might be 8,000-15,000 characters; a 45-minute one might be 4,000-7,000. NEVER truncate. NEVER list bullet points for the main body (only acceptable inside a section, sparingly).

8. **Hard ban.** Do not include a "Key Takeaways" list at the top, do not include numbered "Story 1 / Story 2" headers, do not produce a TL;DR. The whole document is the summary.

9. **Open with a one-sentence framing.** First line: "Here are the takeaways from this <Guest Name> interview (<one-clause description>):" — then blank line, then your first bold heading. The bracket clause should be ARC-FOCUSED if the guest has a memorable trajectory or signature project (e.g. "went from X to Y, originally built Z"), not just a CV-style title. If neither fits, fall back to title + signature contribution.

11. **Let Ryan in, sparingly.** Where the transcript shows Ryan reacting, framing, or pushing back on something noteworthy ("Ryan notes the agency...", "Ryan asks why..."), occasionally surface that — once or twice across the whole summary, not as a tic. Pure-monologue summaries miss the dialogue texture.

12. **Heading discipline.** Bold paragraph headings should be ≤ 12 words and ≤ 80 chars. Trim subtitles after a colon when the colon would push past the cap.

13. **Quote density.** A good summary of a long episode contains roughly 8-20 short direct quotes (5-25 words each), scattered through the sections — not clustered. The guest's voice should be audible in every section. Use double-quotes; don't markdown-italicise the quotes.

14. **Bold vs italic.** Use **bold** for the guest's coined frameworks, playbooks, principles, or system names ("**metastable failure**", "**Magic Pocket**", "**75/25 practitioner-to-communicator**"). Use *italics* sparingly for a single emphasized word inside otherwise normal prose ("the *agency* this early", "his only employer *post-PhD*"). Don't bold ordinary nouns.

15. **Avoid attribution tics.** Don't stack five sentences that all start with "He says / He notes / He stresses / He warns." Vary: lead with the claim, drop attribution when context is obvious, occasionally embed the speaker mid-sentence. Direct prose narrating what was said is fine without "he says" every time.

10. **No capstone in this pass.** Pass 2 will add the closing synthesis paragraph; you focus on the body.

Coverage rules:
- Every project, team, role, decision, principle, and named framework the guest discusses must appear somewhere.
- If the guest disagrees with conventional wisdom, name the disagreement.
- If the guest cites a specific person, book, paper, tool, or company as influential — keep the citation.

Write the summary now based on the transcript that follows."""


P2_SYSTEM = f"""You are doing a polish + completeness pass on a podcast-summary draft. You receive (a) the full transcript, (b) a Pass-1 draft summary. Your output is the FINAL summary — a new, improved version that subsumes the draft.

{_example_block()}

What this pass does:

1. **Read the transcript and the draft.** Identify everything in the transcript that's missing or thin in the draft. Common gaps: a specific number/name dropped, a named framework not bolded, a section too short, a striking quote not quoted, a project mentioned but not narrated.

2. **Produce the FULL improved summary.** Do NOT produce a delta or a list of changes. Rewrite end-to-end, incorporating the draft as your starting point and folding in the missing material.

3. **Add the capstone paragraph at the end.** Three-to-five sentences, after a `---` horizontal rule. It should name cross-cutting themes by referencing the GUEST'S OWN coined frameworks, playbooks, or named concepts that appeared earlier in your summary — e.g. "His **X playbook** echoes his **Y stance**; both come from..." — not generic abstractions like "simplicity" or "real problems." The capstone is a synthesis through their specific vocabulary. It can reference other Peterman Post guests ONLY if the input transcript explicitly names them.

4. **Preserve the user-style rules** that governed Pass 1:
   - Open with the "Here are the takeaways from this <Guest> interview (<one-clause>):" line.
   - Bold paragraph headings (not `##`), chapter-by-chapter, interwoven story+lesson.
   - Inline quotes in double-quotes; bold named concepts; preserve all specifics.
   - Light editorial voice via parentheticals.
   - No bullet lists for main body; no Story-1/Story-2 numbering; no "Key Takeaways" lead-in.

5. **Tighten where the draft is wordy, expand where it is thin.** Comprehensiveness > brevity, but every sentence should carry weight. Strip filler like "the guest emphasized that..." — just state what they said.

6. **Length envelope.** Roughly the length the material justifies. Long episode → 8-15k chars. The capstone adds ~300-800 chars.

7. **Hard ban (same as P1):** no TL;DR, no numbered story headers, no "Key Takeaways" upfront, no bullet-list main body.

Write the improved full summary now."""


def p1_user(article_title: str, transcript: str) -> str:
    return f"""Title of the post you are summarising: {article_title}

Transcript follows. Write the chapter-by-chapter narrative summary per the system rules.

--- TRANSCRIPT ---

{transcript}
"""


def p2_user(article_title: str, transcript: str, p1_output: str) -> str:
    return f"""Title of the post: {article_title}

The Pass-1 draft is below. Audit it against the full transcript that follows, then produce the FINAL improved summary (rewritten end-to-end, ending with the capstone paragraph after a `---` rule).

--- PASS 1 DRAFT ---

{p1_output}

--- FULL TRANSCRIPT ---

{transcript}
"""
