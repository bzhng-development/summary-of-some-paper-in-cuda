Here are the takeaways from this Ryan Peterman post (a practical guide to writing self-reviews that survive the perf calibration room):

**A bad self-review, modeled from Ryan's first attempt**

Ryan opens by showing a self-review he wrote early in his career, structured as a simple bulleted list of three projects with sub-bullets describing the work. The bullets are activity-focused: "Created API to retrieve higher-quality video," "Altered existing pipeline," "Was the main IC from the server side that built the entire system end to end," "Verification on encodings… shows no problems." Even where outcomes appear ("Upload quality improved by X%"), the significance is left unexplained. The formatting is dense, and the ordering doesn't prioritize the most collaborative or highest-impact work. He uses this as the foil for everything that follows — a review that describes effort without translating it into the language of impact that a room of skim-reading managers can absorb in roughly three minutes.

**Less is more: cut activity, keep results**

The first principle is stripping sentences that describe the *work itself* rather than its consequences. Ryan points to lines like "Verification on encodings produced from the new pipeline vs old pipeline shows no problems" and "This includes the new video pipeline, the new data model, and the new delivery code." These tell a reader what you did, not why it mattered. Their presence dilutes the signal a manager needs to extract quickly. The goal is short, dense sentences that explain results — not a task log. As he puts it, "aim for short sentences rich in details that explain your work's results, not the work itself."

**Know your audience: a room of managers with no context**

The calibration room contains managers who lack deep familiarity with your domain. Your self-review must bridge that gap by anchoring impact in metrics and milestones they *can* evaluate cross-team. In the bad example, "Built out a new video encoding pipeline using new partner team system" is followed by rollout percentages but never explains *why* the migration matters. Ryan's rewrite frames the same work around a shared-roadmap milestone, cites the number of engineers involved, and explicitly links the migration to preventing specific severe past incidents (with links). Similarly, the "X% upload quality improvement" gets retrofitted with the parenthetical "(30% of team goal)," instantly communicating magnitude. The rule: **translate impact into metrics and milestones managers already understand**, and supply the one sentence of context that makes a number legible.

**Managing limited attention: lead with your strongest story**

Because calibration readers skim, order matters. Ryan's original review listed the higher-quality API first, but the new video product (point 2) was a larger cross-team collaboration with more demonstrable impact. His advice: "put your most impactful work first to make sure they see it." In the revised version, the new video product ships first — framed as shipping one month ahead of a shared goal with N engineers and no regressions, driving X% more video creation. The higher-quality API and encoding pipeline follow, each condensed to one outcome-dense line.

**The manager's role: your self-review as raw material**

Ryan closes with a meta-point about the perf process itself: your self-review isn't the final packet. Your manager condenses it, adds additional signals, and refines it into the artifact that enters calibration. If you're great at writing self-reviews, "your manager will reuse a lot of what you wrote to represent your work." This creates a pragmatic heuristic for what to include: when in doubt, leave it in. Your manager can cut less impactful items more easily than they can reconstruct missing context. The self-review is your chance to shape the raw material before it gets compressed.