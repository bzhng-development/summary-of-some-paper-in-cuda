Here are the takeaways from this Ryan Peterman essay (on how the plummeting cost of automation via LLMs is reshaping his entire relationship with his computer):

**The old automation calculus is dead**

The decision to automate a task used to be a straightforward cost-benefit equation: automate if the benefit exceeded the cost of building it. For most people, that meant automating only frequent, repetitive tasks—the kind you’d encounter over and over, justifying the upfront scripting investment. One-off nuisances weren't worth it; you'd just muscle through them manually. But a "fundamental part of this equation has completely changed in the last year."

**What changed: code generation and agentic tooling**

The trigger is the arrival of code generation models and agentic tooling like Codex or Claude Code—which Ryan names while giving the aside "(not an ad)." These tools have made the *cost* of automating something collapse toward zero. Ryan’s new principle is stark: "Now, automating work is almost free. I automate just about everything if it’s easy to access from the terminal." The threshold has dropped so low that the old calculus no longer applies. He's not just automating more of the same kinds of tasks—he's automating an entirely new *class* of task he would have never touched before, and the scripts are so cheap he generates them "more often than I do these small local tasks myself."

**Podcast transcript rescue**

His first example: podcast transcription is inaccurate and needs fixing, and LLMs can correct the errors. The catch is that an LLM can't ingest a two-hour transcript in one shot. So he had an AI generate a script that chunks the transcript into pieces and feeds each piece to an LLM API sequentially. The result is a single terminal command that processes the whole file. This didn't require deep scripting knowledge—just describing the problem and letting the tool produce the solution.

**Stitching corrupted video chunks from raw files**

A more dramatic one-off came when his remote podcast recording software failed to upload properly. The uploads broke, but the guest dug through his Chrome folder and discovered all the raw video chunks—hundreds of numbered files—and sent Ryan a zip. Ryan had an AI generate a script to list and sort the files, then used ffmpeg to stitch everything back into a single video. He notes that even though this is not a task he expects to *ever* do again, "generating the script was still easier than figuring out the ffmpeg command by hand." In the pre-LLM world, he would have spent ages reading man pages and debugging arcane ffmpeg flags for a one-time fix. Now the script was the path of least resistance.

**Bulk-converting notes to Markdown for Obsidian**

A third example: migrating all his personal notes into Obsidian, which expects Markdown. His notes were scattered across Google Docs, Apple Notes, and Google Keep—each with its own export format. He dumped all the exported zip files into one directory and had an AI write three separate scripts, one per format, to convert everything to Markdown. Again, a task he won't repeat, yet "generating a script was the fastest way." His candid admission: "In the past, I wouldn’t have done the migration, but now it’s so easy I figured why not." That phrase—"why not"—captures the entire shift. The cost has dropped so low that *curiosity* and *tidiness* now justify automation, not just hard-nosed efficiency.

**A new relationship with the computer**

The cumulative effect is a qualitative change in how he uses his machine. "This has changed how I use my computer," he writes. He now reaches for AI-generated scripts more often than he does small local tasks manually. The tools are not perfect, but for these small, targeted problems they are actually great: "LLMs are likely to get it right the first time, and even if not, being wrong doesn’t matter much." A hallucinated ffmpeg flag on a local script won't break production. The downside is minimal, so the upside is almost pure. He notes this amidst the broader AI hype—"this is one real use case that already saves me a ton of time."

**Beyond time savings: doing what was never worth doing**

The subtlest payoff isn't saved minutes but expanded possibility. "Not only do these agentic tools save me time, they let me do things I wouldn’t have done before because they weren’t worth the trouble." Migrating a decade of scattered notes, recovering a corrupted video from raw browser cache, cleaning up a two-hour transcript—these aren't optimizations to an existing workflow. They're entire projects that would have been discarded as impractical. The zero-cost automation thesis is that **the automation decision is no longer about frequency; it's about desire**. If you sort-of want to do a thing and it's terminal-accessible, you just do it. The cost is "basically zero," so the only remaining question is whether you feel like it.

---

Ryan's central reframing—that the automation decision has shifted from a frequency-based calculus to a desire-based one—flows directly from his observation that the **cost of automating** has collapsed. The three examples build a nested argument: the podcast script shows the new unit economics (chunk-and-API is now trivial), the ffmpeg rescue shows that even a *one-time* task is cheaper to script than to do manually, and the Obsidian migration shows the psychological flip from "not worth the trouble" to "why not." Together they form his **zero-cost automation thesis**: when the downside of a wrong script is negligible and the upside is pure leverage, your computer becomes a place of expanded possibility rather than constrained toil.