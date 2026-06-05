<sup>Source: <https://www.developing.dev/p/stanford-phd-ai-researcher-and-quant></sup>

Here are the takeaways from this Nimit Sohani interview (Stanford PhD who moved from AI research to quant at Citadel and back to AI as a researcher at Cartesia, where he works on voice AI and state space models):

**Do you need a PhD for AI research or quant?**

The roles aren't formally gated—plenty of people do AI research without a PhD, and plenty enter quant from undergrad—but the PhD changes the "type and shape" of the role and makes getting in much easier. In AI research, without a PhD you tend toward engineering-heavy work (training infrastructure, eval pipelines, data processing). With one, you're more likely to do "pie in the sky" architecture design and fundamental exploration that may not pay off short-term. He stresses the credential opens doors at the filtering stage, but the deeper value is the skill set developed: **90% of the battle in research is finding the right problems**—scoping them to be tractable, convincing others they matter, then executing. "That was a big learning process for me during the PhD—that sort of research taste and problem selection."

Industry skews heavily applied and makes that exploratory first-principles muscle harder to build. But if your only goal is to be an AI researcher and you don't care which flavor of work you do, "a PhD is definitely not necessary." It's most valuable if you're still in career-exploration mode and want to find a problem that genuinely draws you in.

**Building research taste: read broadly, grow organically, start small**

His method for developing research taste is simple in principle and hard in practice: read as many papers as you can—abstracts count, you don't have to go end-to-end—and talk to people. His primary discovery surface is X (Twitter): he followed Stanford peers and professors whose papers he'd read, then whenever they tweeted or liked a paper he found interesting, he'd click through and follow everyone tagged. His feed is now "mostly machine learning papers and pictures of cute animals." He grew his follow list organically: "anytime they tweet a paper, they like a paper... if it's interesting to me, I just click on that and I follow all the people tagged in or associated with that work."

The other piece is staged maturity. Early researchers should attack small sub-problems where progress is likely—extending a known method to a slightly different case. As you mature, you can tackle bigger, genuinely novel ideas. **Trying to skip those steps is "generally inadvisable."**

**Why he became a quant after the PhD**

He'd been doing AI research for four to five years already and was curious about other careers that married mathematics and computation. The three he saw were ML research (which he knew), quantitative finance, and quantum computing (too small a domain at the time). He interned at Citadel Securities the summer before graduating, liked it, and joined full-time as a quantitative researcher. "It was refreshing in some ways. The PhD is a grind, you can burn out at various points. It was a fresh set of problems, totally different environment."

On the legendary intensity of quant firms: his experience was actually a good work-life balance. Traders cluster around US market hours, and that rhythm trickles down culturally so people "don't take their work home too much." He notes the AI field is now more competitive in hours-worked: "one of the ways you gain comparative advantage is by outworking your competition, and that's what happens in practice at a lot of places."

**What quants actually do**

The job varies enormously by firm type (hedge fund vs. market maker), desk (alpha generation, monetization, risk, data analysis), and company. Some quants do pure strategy discovery and backtesting; others only monetize existing alphas; risk quants design metrics to cut risk without cutting profit. "The thing that unifies all of them is having a strong math background." The backbone is stochastic calculus, but the field pulls in numerical optimization, interpolation, numerical linear algebra, and increasingly deep learning—some firms now run their own LLM research arms. He personally did "a ton of coding, mostly in C, also some Python." The split between quants and software engineers varies: at Jane Street, "quants" is a tiny label and traders are highly technical; at Citadel, the firm is more quant-forward and the roles overlap considerably.

**The cultural chasm between quant finance and tech: secrecy, comp, and garden leave**

The biggest shock moving from quant back to AI was openness. "In finance, even firms with a reputation for being open are actually quite secretive." Alphas only work if they're unknown; the more people trade a strategy, the less profitable it becomes for anyone. This creates internal silos—pod structures where teams don't share ideas because uncorrelated returns reduce wipeout risk. "Hearing people in tech talk about what they're doing in a very open way—it was like, wow, you're just going to tell me that for free?"

Compensation is opaque and unstandardized—there are no IC5 pay bands. It's a function of the firm's year, the team's year, your personal P&L (for alpha-side roles), seniority, and tenure. "In the top 1% of either quant or AI, you're doing very well. People are making NBA-player salaries." But higher alpha exposure also means higher variance and job risk.

Non-competes (garden leave) are standard: when you leave, the firm can require you not to work for a competitor for anywhere from zero to two (sometimes three) years, during which you're paid. The logic is that your specific alphas will be irrelevant by the time it expires. It also creates a strange incentive where senior people effectively get a well-compensated forced sabbatical. He calls the norm "six months to two years."

**Job security is U-shaped**

Senior quants get expensive because compensation rises with tenure. "Even a good quant can stop being worth it after a while, whereas earlier-career quants might be very good and not command as high a salary." This produces an "inverse parabola" risk curve—strong performers can be cut when the cost equation flips. On the alpha side, performance is brutally transparent: if you're not making money, it's clear. For traders, it's "even more brutal" since outcomes are continuously measurable. The quant culture includes "trimming the low performers" and an up-or-out dynamic.

He also shared a warning story: firms don't mess around with insider trading. Quants have trade pre-approval requirements, and people who try to get around them via WhatsApp groups telling friends what to buy "get found out, you'll get fired, there'll be lawsuits, you can even go to jail." Similarly, violating non-competes by taking strategies to competitors invites elite legal teams.

**Quant firm tier list—the mythical and the elite**

Rentech is "one of the sort of mythical firms in the space" with insane 20-30-year historical returns—"the gold standard." Jane Street, Citadel, Jump Trading, Hudson River are generally very well-regarded. Then there are elite smaller ones similar to Rentech: TGS (Southern California), XTX (a newer firm), Radix (another newer firm)—"smaller, more secretive, less well known, but still very, very excellent returns."

**Leaving Citadel for a voice AI startup**

He left after a couple of years, feeling his growth curve was beginning to taper. He saw the AI landscape exploding post-ChatGPT and heard the founders of Cartesia—all from his PhD lab (Chris Ré's group at Stanford)—were starting a company. He knew them well; Albert Gu was a good friend. The draw was a combination of technical growth, the chance to shape an early-stage startup, and a deliberate risk-profile shift: "When I graduated my PhD I was more risk-averse. Quant was a stable, lucrative opportunity. Now that I'd established some stability, I thought it was an opportune time to take a risk."

**What Cartesia builds and who they're fighting**

Cartesia is a voice AI company: text-to-speech is the flagship, with speech-to-text and voice agents expanding the surface. Their main competitor is ElevenLabs, which had an ~18-month head start. Cartesia's differentiator is latency—"you can't afford a second pause between turns of a conversation; that breaks the illusion"—and they're pushing toward end-to-end speech models rather than the standard cascade (ASR → LLM backbone → TTS), which loses naturalness at every handoff.

Winning means not just one metric but a portfolio: transcript fidelity (surprisingly hard across languages and special characters), naturalness (often valued more than fidelity), voice cloning, accent localization, controllability over speed and emotion. "Switching costs exist even in AI. If you can conclusively show you're better in every way, at some point it becomes hard to argue for not switching."

**Why a startup over a big lab: challenging orthodoxy**

Big labs have infinite compute and talent, but that breeds conservatism. "They can be more averse to out-of-the-box ideas and more susceptible to groupthink." At a startup you can be nimble and strategic about challenging accepted wisdom. At the time Mamba came out, the prevailing view was that sequence modeling was solved—just scale transformers further. Albert Gu showed state space models could win on efficiency and even raw quality for certain problem classes. More recently, Cartesia published work on **H-nets**—learning tokenization boundaries directly from raw characters instead of a separate tokenizer pipeline, getting better performance by questioning a foundational assumption. "That's the kind of thing—challenging accepted ideas—that appealed to me."

**State space models vs. transformers: a quick primer**

Transformers store a representation of every token in the KV cache, so memory and compute cost grow linearly with sequence length. State space models (SSMs) compress that information into a fixed-size state—like a brain versus a database. The human brain doesn't store an unbounded amount of context; it processes and keeps information in a fixed-size state. "You can simulate having an unbounded state via use of external tools like writing stuff down, but the core primitive remains fixed." For recall-heavy factual tasks, pure SSMs lag because transformers' exact in-context recall is genuinely useful. For other tasks, SSMs scale as well or better at the same parameter budget. The cutting edge for text is now **hybrid models** interleaving SSM and transformer layers (Nvidia, the latest Qwen models).

For audio, SSMs are practically a free lunch. Audio frames (10–100 ms) contain very little information per timestep—one frame barely differs from the next—so compression as an explicit inductive bias actually improves both quality and inference speed. "You cannot design your architecture independently of your data. Co-design and thinking about multimodality from a fundamental level drives a lot of the work we do here."

**The product-plus-research thesis**

Pure research startups without a product path make him skeptical. Big labs already have more resources for that. Conversely, pure product companies built on other people's models lack a moat—GPT-4 upgrades obsoleted a generation of wrapper startups because the base model itself swallowed the post-processing. Being at the intersection is powerful: real customer problems drive research toward modeling-level fixes rather than bandaids. "Cartesia is first and foremost a product company, but we believe building the best products requires solving fundamental research problems." Having control over the models is essential; it lets you fix issues "from the ground up at the model level itself."

**Advice for SWEs trying to move into AI research**

His philosophy: **build fundamentals so deep that opportunities come to you rather than the other way around.** Get as good as you can at coding and math; read papers voraciously; develop math intuition. At big companies, pivoting teams is harder and you can get siloed—switching companies or teams is sometimes the only path. A master's in AI can serve as the qualification that makes a lateral switch possible. At startups like Cartesia, people have successfully transitioned from SWE to research roles organically, because everyone knows everyone and you can demonstrate aptitude over time—"this has actually happened in Cartesia itself, with a lot of success." But you still need some evidence of the skill set, whether self-built or credentialed. "There's gotta be something behind it."

**On regret and advice to his younger self**

"I often overthink things and I've spent a lot of time regretting past decisions that turned out not to matter. I kind of regret the amount of time I spent regretting other things." His meta-lesson: minor setbacks happen, beating yourself up is unproductive, and it makes nobody feel good. "Don't sweat the small stuff." To his younger self entering the industry, he'd say: focus entirely on deep technical skills, don't spread yourself too thin, and spend your work time on the capacities you want to leverage day-to-day. "It's a simple recipe that's very hard to follow—kind of like being healthier: exercise and eat right. It is that simple."

---

His career arcs through AI research, quantitative finance, and back again all converge on a single principle: **90% of the battle is finding the right problems**—whether that's a tractable research question, an unexploited alpha, or a product feature worth modeling from the ground up. His conviction that you should **build fundamentals so deep that opportunities come to you** explains both his smooth lateral moves and his skepticism of shortcuts; the PhD wasn't a credential as much as it was the forced practice of problem-selection and research-taste development. The same instinct shows up in Cartesia's technical strategy—the bet on state space models, the **H-nets** insight that you can learn token boundaries from raw characters, and the insistence on co-designing architecture with the data modality—all are forms of questioning orthodoxy where others assume the prevailing recipe is solved. Even the cultural contrast between quant secrecy and AI openness underscores the same meta-point: in one world, edge comes from hiding the problem; in the other, edge comes from sharing it widely and working on harder ones.
