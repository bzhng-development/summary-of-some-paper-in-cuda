# Learnings & Principles

## On Systems Design & Architecture

**Simplicity is the hardest thing in systems.** James argues that simple systems are far more difficult to design than complex ones, but they're what scale — not just in queries per second, but in surviving five years of feature additions and changing requirements. "To the untrained eye, a simple system can seem obvious." The best compliment is when people say "isn't that the obvious way of doing it?" — because it wasn't obvious when you did it, and no one else was doing it.

> "Simplicity is scalable. And yes, simplicity is scalable in terms of numbers of queries per second. But what I really mean about scalability is you can take a simple system and have it run for five years, have people work in it for five years, have all sorts of features added to it, and have requirements changed because the company realized the product didn't work the way it wanted to work and it wants to change things. It still stands the test of time, whereas a complex, over-optimized system will not."

**Design for validation, not sophistication.** When building the storage system at Dropbox, James used a simple MySQL cluster (1,000 nodes) indexed by block ID rather than a distributed hash table or Patricia trie. The reasoning: "If I want to validate what happened, if I want to check all the data is where it's meant to be, I just walk over the table and check." Every new hire from academia would suggest a more sophisticated approach, but James argues that sophistication optimizes for the wrong thing — what matters is what happens when the system doesn't work. "Designing for validation is very important. Designing for understanding is very important."

**Don't over-architect for problems you don't have.** This is drilled into the team at Convex: "Everything exists for the why. Don't build a fancy load balancer unless it's needed." (They do need one now, and they're building it — but the principle is to start with the problem, not the solution.)

**Performance is about eliminating coordination points, not raw hardware speed.** From his Granola work: "No one has particularly faster disks or memory than anybody else. What really matters to performance in a large-scale system is eliminating points of coordination. It's how to allow systems to progress without having contention between parties."

**Tiered architectures based on workload understanding create real efficiency.** At Dropbox, James built two clusters: one for temporary/hot storage (storage-inefficient but access-efficient) and one for cold storage (more static, more efficient, bulk-written). This was possible because they deeply understood their workload — access patterns decay predictably, block sizes are known, and writes and reads can be separated.

**Most companies should not do active-active multi-homing.** Despite it being an aspiration for many engineers, the latency cost (roughly 60ms for synchronous cross-US commits) is prohibitive for most applications. "If Amazon's down that day, your company will be down. That's a shame, right? But by avoiding that complexity, you're going to be able to move much faster and build a much better product." Systems are all about trade-offs.

**Engineering is science with constraints.** James defines engineering as solving problems in the presence of resource constraints. "I'm not particularly interested in constraint-free environments. That's art. I like craft and engineering. The more visceral and difficult the constraints, the more fun that is for me."

**Complexity destroys systems over time.** "The tough thing about distributed systems design, especially LLM-augmented distributed systems design: just because something works doesn't mean it's maintainable over a long period of time, doesn't mean it's understandable, and doesn't mean it's cleanly architected and abstracted."

## On Organizational Culture & Leadership

**Orient teams around missions, not systems.** After Magic Pocket shipped, James renamed the team from "Magic Pocket Team" to "Storage Team" — requiring renaming email addresses, Slack channels, repos. He explains: the responsibility is to solve the storage needs of the organization, not advocate for Magic Pocket specifically. If S3 ever becomes the better choice, it should be the storage team's job to advocate for moving back. Otherwise, the system becomes tied to team identity and career security.

**Inertia is the enemy of good decisions.** James's most impactful job at Dropbox was shutting down projects. He'd talk to teams privately and ask: "Do you believe in what you're doing? Does this make sense?" Teams would often admit they didn't really know. But inertia is so powerful — "this whole desire to not get in trouble, to just keep doing what you were previously doing is so strong, and talented people can end up doing things that don't make a lot of sense."

**Prioritization, not yes/no.** When engineers complained about inefficiency, James would ask: "Do you think we should solve this problem right now? Because if we should, let me know the team to take some engineers off and the product to shut down." Often the answer was that nothing else was lower priority — at which point you just accept the constraint. "There's no point in being angry or upset that we're not doing the right thing all the time." Nothing is a yes/no question; it's a prioritization question.

**Don't lead by example — it's passive and ineffective.** James initially tried to model on-call ownership by always responding first, writing incident reports, jumping on bugs. But from the team's perspective, this looked like either "that's James's job" or "James likes doing those things." He now believes leadership requires explicit conversations, not passive modeling: "You can't just act a certain way in front of people and wait for them to copy you."

**The oversight/accountability slider.** James conceptualizes leadership as a slider between oversight and accountability. When someone is new/junior, you check their work (oversight). As they grow, you must deliberately dial down oversight while dialing up accountability through explicit conversations: "Okay, cool, you've got this project. Let me know when it's going to get done. Next Thursday. What's the plan? How are you going to know it's correct? It's on you." You can't skip this step; you can't just stop micromanaging and expect ownership to develop.

**Paper cuts vs. losing an arm.** When giving people accountability, let them make small mistakes (paper cuts) — these are growth experiences. But don't let a junior person design the replication system. Part of the art of leadership is figuring out the right level of autonomy for each person and task.

**100% alignment on "why," then trust on "how."** Most organizational conflict happens because well-intentioned people are debating "how" when they don't agree on "why." If one team thinks the priority is features and another thinks it's reliability, they'll naturally do very different things. James believes everyone needs alignment on why, then largely trusts the team on implementation.

**You can't change someone's mind in a meeting.** James has to tell this to many senior engineers. Changing someone's mind "involves going through a complex series of neurological processes that don't happen live in front of ten people in a meeting." If you force someone to agree, their ego gets bruised. Instead: identify the disagreement, provide information, then let it sit. Let them reflect and come back a week later.

**Engineers should go into management because they have to, not because they want to.** "I feel like most people should not want to be managers, and I sometimes think it's a bit of a red flag if someone wants to be a manager too much." James only went into management when it was necessary, and bounced back to engineering as soon as someone else could take over.

**Don't go into management too early.** Ideally get to staff engineer first. If you go into management before you're an excellent technician, "it will limit your ability to influence strategy later in your career." You'll be stuck doing people management without being able to evaluate technical work or push the team toward excellence.

**Everyone needs to become the best version of themselves.** There's no single archetype for a senior principal engineer. James's brand is "strategic collaboration, simplicity, abstraction" — and his coding fell off a cliff. Other engineers are deep science/hardcore problem-solving types. The mentoring job at senior levels is helping people find their strengths and become more "spiky."

**Effectiveness matters more than being right.** "It doesn't matter how smart you are, it doesn't matter how right you are; are you effective?" At senior levels, most hard problems happen with a team of 10-20 people. If you can't change people's minds or convince teams, you're "kind of useless" regardless of how correct your technical analysis is.

**Capitalism of interpersonal behavior.** James draws an analogy: capitalism rewards success, not intention. If you didn't manage to convince someone, it doesn't matter how well-intentioned you were. This applies to changing minds, getting teams to adopt your API, etc.

## On Career Growth

**Stay long enough to own the consequences of your decisions.** "If you're not in a job for three years, you're not going to see whether your decisions were good. You can get more money as a junior engineer, but you cannot become a very talented senior engineer without being around for long enough to own the consequences of your decisions." It's like playing a basketball game and leaving before it's over.

**The career ladder isn't about getting better at programming.** James thinks his programming abilities went down from level four onward. What changes is the scope you care about. "At a certain point, you're just thinking about what matters most at the company for the next five years." IC 6, 7, 8 engineers aren't necessarily the best programmers — they're better at having a broad perspective on decisions.

**Working with the best people is the highest-leverage career investment.** If the choice is 20% more money vs. working with the best people in the world, pick the people. "That's going to set you on the ship to success." (He caveats: there's no shame in wanting a regular job and chilling, but if you want to maximize growth, maximize your skills and environment.)

**Wisdom requires synthesis, not just knowledge.** "You can go ask ChatGPT how two-phase commit works and it will give you a pretty good answer. But growth as an engineer does require wisdom, and wisdom only really happens when you synthesize it." Facts put into practice, then internalized.

**Do the reps yourself.** In the AI era, it's easy to fall into a passivity trap. James uses the gym analogy: "If you went to the gym and picked up a heavy weight and let the robot pick it up for you, you're also not really growing." He recommends trying to solve problems yourself first, then checking with an LLM. Spend some time "in the intellectual wilderness of not being able to solve a problem and struggling."

**Careers are long; don't max out in three months.** "There's a feeling right now, oh my God, AGI has come and better max out my growth in the next three months. Well, guess what? You ain't going to do it. It's not going to happen." Engineers can grow for 20+ years.

**Don't over-optimize for promotion early.** Junior engineers often over-optimize for promotion and salary rather than investing in themselves. To a certain level of seniority, you've "made it" and money is fine. The real luxury is enjoying what you do every day.

**Tech tabloidism is noise.** "It's like reading about Beyoncé, but you're a nerd, and so you're reading about Jensen." The new model drops, the new tool appears — it's fine to miss it. If it turns out to be important, you'll just use it later. "Just watch out for tech tabloidism. It doesn't matter. Just be building stuff. Just do real work."

**A PhD is training to be a researcher, not college continued.** Many high-achieving students think a PhD is just more learning. It's not — it's training for research. For most people who want to be software engineers, they shouldn't do it. But the developmental experience of facing a problem no one in the world knows the answer to is "really valuable for all engineers."

**Academia is for advancing knowledge; industry is for solving problems.** James gravitates toward solving problems because he finds it "a more comfortable environment." He didn't like academia's structure of making up a problem, making up a solution, and then trying to convince people it was good. "I just wanted to build it and see if it was good."

**Invest in personal life, not just career.** James's primary career regret is underinvesting in his personal life. "I've been on call my whole career. I've carried a laptop almost every day. There are many dinners, parties, and events I've had to skip, and there are people in my personal life who have suffered as a result." He'd advise his younger self to do more vacations and have more balance.

## On Transactional & Distributed Systems

**Transactions are one of the most incredible abstractions we've invented.** They allow us "to manage probably the most difficult problem in computer science, which is concurrency."

**Two-phase commit with two-phase locking can be low performance and high risk.** It blocks systems for the transaction duration and creates a dependency on another node — you're "blocked waiting for another node to return."

**Erasure coding can be faster than simple replication.** If you need 6 out of 9 fragments to reconstruct, you ask all 9 and return as soon as the first 6 respond. You can construct encoding matrices to optimize for this. In practice at Dropbox, they'd often keep a full copy on a single disk for fast access and ensure data was served from a region close to the user's home.

**Durability of 24 nines means the universe will be extinct before data is lost.**
This requires spreading fragments across racks, rows, power feeds, drive manufacturers, drive eras, and regions — taking correlated failure patterns into account.

**Running multi-exabyte storage on physical hardware means decisions about power per rack matter.** "You have a rack of hardware; there's a power distribution unit. At the top of that rack, it has a circuit breaker which can handle a certain number of amps. We would have to figure out how many amps are required for that rack based on access patterns." Getting it wrong meant messages from the data center team saying racks were running too hot.


# Opinions & Hot Takes

**Google "ruined" systems research (with affection and respect).** When papers from Google and Amazon started being about systems that powered Gmail rather than just ideas, program committees began expecting realistic benchmarks that grad students couldn't produce. "I think it did, in some respects, obscure intellectual ideas." James prefers papers about ideas, but also sees value in papers about systems. The problem was the shift in expectation — academic program committees started demanding industrial-scale validation that academia couldn't provide.

**Most companies should not move off the cloud.** "If you want to build a more efficient storage system than Amazon, you have to have a supply chain team that's working with Western Digital and Seagate, constantly negotiating on prices of disks and buying shipments at certain times, along with capacity teams and data center teams." Moving off the cloud only makes sense with very small/fixed requirements or very heavy investment. "Most people should focus on their applications."

**The promotion/incentive system at large companies is broken and forces complexity.** "It almost angers me." When promotions get rejected because work wasn't complex enough, it creates perverse incentives. James contrasts this with a startup where "the goal is to build the system, have it work, have the users like it, have it grow, and everyone gets rewarded and celebrated for solving the problem." At scale, the distance from actual outcomes creates artificial incentive structures.

**People who chase artificial goals (OKR green checkmarks) are missing the point.** "Who cares about your OKR plan unless it solves the problem?" This is a symptom of being too many layers removed from actual impact.

**"Maybe your manager doesn't understand you if they haven't spent enough time developing technical skills."** Non-technical managers can't evaluate their team's work or know if the team is doing well. This contributes to cynical attitudes where engineers feel misunderstood.

**Coding and engineering are very different things.** Even if coding gets commoditized by AI, engineering — conceptualizing problems, breaking them down, designing clean solutions — remains valuable. "I still do think that there is a very, very promising role for human beings in engineering."

**The "don't learn anything because AGI is coming" take is ludicrous.** "I'm not sure if you've seen people say these ludicrous things, like, oh, maybe in the future, not knowing engineering will be an advantage because you won't have biases and you'll just use Claude. I think these are ludicrous statements." Software engineering is an intellectual discipline that trains your mind. The "permanent underclass" narrative is "not an instructive attitude" — there's not much you can do with that information other than feel bad.

**Stop the AI doomer narrative.** "I really wish my peers and my cohort would stop it with the real doomer, human elimination kind of narrative." The exciting framing is: "Look at all this new cool stuff we can build. Look at the ways we can make people's lives better."

**Agentic coding tools are not good at simplicity.** "They're not the best at building simple systems. Simplicity is still the domain of human beings for now."

**The labs are all desperately hiring senior engineers.** Despite the AI hype, "no matter what they say, they're still hiring engineers. Desperately hiring engineers. Really, really aggressively hiring engineers." Even Anthropic still does whiteboard coding interviews.

**Python gets a bad rap for I/O-bound workloads.** "It's pretty good at I/O, but obviously not great for concurrency, not great for memory management, and very hard to refactor."

**Go is great for proxies but the GC runtime is dangerous for storage systems.**
Memory unpredictability with Go's garbage collector was a major problem at Dropbox because OOM on a storage node looks like a disk failure, which triggers re-replication, which can cause cascading failures.

**None of the mainstream databases are that great.** "AWS is a fine tool. PostgreSQL is a fine tool, although none of the mainstream databases are that great, frankly." But the real issue is they don't make problems go away — they still require reasoning about state management, concurrency, polling, data sync.

**The abstraction floor needs to rise.** "The world is and has been overdue for a new abstraction one level up the stack." This is what Convex aims to provide — making problems go away rather than just giving developers tools and leaving them to figure out architecture.

**Meetings generally aren't for decision making.** True decisions, especially changing someone's mind, happen after reflection, not live in front of ten people.

**Hustle culture and performative laptop-at-the-bar photos aren't real.** "I see that with all the 996 stuff and this kind of performative photos of being in a bar with a laptop, and I'm like that's not real." Working long hours isn't the goal — doing cool stuff is.

**The 22-year-old billionaire story isn't real, repeatable, normal, or healthy.** "Just ignore that story."

**Being around for the consequences is structurally incentivized but underrated.** Job hopping can get you more money short-term, but you can't become a senior engineer without owning the outcomes of your decisions over years.

**There's value to an organization in having hard technical problems.** "If you have a company with extremely hard technical challenges, you can attract engineers who like working on those hard technical problems." When those engineers solve them, they cycle off and apply their skills to other systems — like redesigning the sync protocol or file system.

**Active-active multi-homing is something most companies should NOT do.** Despite many teams reaching out for advice on how to adopt it, James almost universally recommends against it. The speed of light is fixed, and the latency cost is prohibitive. The reality for most companies: if US East is down, your company is down, and that's okay given what you gain in velocity by avoiding that complexity.

**Most people shouldn't do a PhD unless they actually want to be a researcher.**
But the developmental experience of facing problems no one knows the answer to is uniquely valuable.

**The 16-hour days early in his career built credibility capital.** Not advocating for it, but "I think it became pretty obvious to people that I cared. This is a guy over here that really wants to do the right thing and cares about the company." That capital gave him the psychological safety to make hard calls later.


# Smaller Anecdotes

**Getting a "slightly nasty" email from someone at AWS during the migration.** As Dropbox moved data off S3, they peaked at 764 gigabits/second of peering bandwidth. Someone at Amazon's network team noticed the unusual read/write pattern and emailed James: "It's super weird that you're doing so many reads and not that many writes." James didn't respond.

**Amazon was a great partner and there was no concern they'd do the wrong thing.** Despite the strained relationship during migration, James emphasizes Dropbox had "only an excellent experience with Amazon Web Services" and still uses AWS. The migration was about efficiency, not adversarial dynamics.

**The prototype for Dropbox's storage system was in Python.** "If you can believe that." It was actually decent for I/O but terrible for concurrency, memory management, and refactoring — and correctness was critical.

**Running pre-GA Go and then pre-GA Rust.** The team built on Go before it was generally available, and then migrated to Rust before it was generally available. "That was a bit of a risky move."

**Operating off draft specs from disk manufacturers.** For the Shingled Magnetic Recording disks and the ZBC instruction set, "the disk manufacturers gave us the draft specs of these new disks, and we were operating off the draft specs and directly controlling the disks."

**The pie chart of costs included sheet metal and screws.** When optimizing at million-hard-drive scale, every physical component matters. The Discotech project (rewrite to Rust + eliminating the filesystem + direct disk addressing) shifted the cost pie chart to be "almost all disks."

**Trampoline: dumping to S3 as an elastic overflow.** To run closer to capacity limits, Dropbox built a system that would dump data to S3 under worst-case scenarios. "Save us a ton of money." They'd test it periodically, but it didn't trigger often.

**Never compromised on durability.** "We had absolute zero, non-negotiable standards; there was no room for negotiation on user durability." They had defcon knobs for CPU, memory, and background processes, but never for data safety.

**FMEA as a giant spreadsheet of every possible failure mode.** It included existential risks like "does someone die?" and "if there's a fire in the data center" — then they would design protections against each one.

**"We're the idiots."** When the Dropbox infrastructure team was 7-9 people and someone from Google would say "someone needs to build this logging framework," the response was: "Well, it's just us. We build it or we don't build it. There's no other idiots out there. We're the idiots."

**James was making the least money on his team at one point.** He was tech leading the team and saw everyone's salary as a technical manager. "I was making the least money, but whatever. It all worked out in the end."

**James wanted to do the Silicon Valley consulting unpaid but had to get $400.**
Hollywood union rules required payment, and the $400 complicated his visa. "I made a grand sum of $400 off that show."

**The show runners cared deeply about technical accuracy.** When building a data center set in a house, they wanted to know exactly what the racks should look like and what diagrams should be on the wall. James thought no one would care, but they insisted it mattered.

**Mike Judge was a software engineer at Lockheed earlier in his career.** "Which a lot of people don't realize." His movie Office Space and Silicon Valley were throwbacks to that experience.

**"RALPH" as an example of tech tabloidism.** A tool/agent that was the biggest thing on Twitter for a month and then disappeared. James uses this as an example of noise that doesn't matter for real engineering growth.

**James works with his hands to wind down.** He has a workshop at home and goes home to make things. "You become an infra person; you become an engineer in all aspects of your life."

**Candidates are getting worse at coding, but whiteboard interviews remain the best evaluation tool.** James acknowledges candidates are getting worse at raw coding (likely due to AI tools), but he doesn't know a better mechanism for evaluating intellectual problem-solving capacity.

**James hasn't read almost any technical book.** He was in academia for a long time so he read papers instead. "Most of my learning has been by doing."

**Enjoying what you do every day is more luxury than a first-class flight.** "The real privilege that we get is to work on something cool."


# Names, Books, Projects, Influences

**Barbara Liskov** → James's PhD advisor at MIT. Referenced multiple times: they worked together on formally modeling consensus protocols, talking on weekends. He also mentions Ryan's prior interview with her and cites her work on abstraction and minimizing complexity as foundational to Convex's philosophy.

**Granola** → James's PhD thesis protocol for distributed transaction coordination using independent transactions and timestamp exchange to avoid two-phase commit.

**View Stamp Replication Revisited** → Paper James wrote during grad school, redefining a protocol that predated Paxos. Influenced companies like **Tiger Beetle**.

**Tiger Beetle** → A company influenced by James's View Stamp Replication Revisited paper.

**Spanner** → Google's globally-distributed database. James's Granola work came out slightly before it and got cited in the Spanner paper, but Spanner subsequently overshadowed it.

**Planet Lab** → Pre-cloud communal academic system for running distributed systems benchmarks. Famously unreliable — James would sleep next to his desk and wake up at random times to check its status and kick off jobs.

**Magic Pocket** → The code name for Dropbox's custom multi-exabyte storage system. Team was originally named after it; James deliberately renamed the team to "Storage" after launch to avoid system bias.

**Discotech** → The "disk technology project" at Dropbox — the Rust rewrite that coincided with eliminating the filesystem and directly addressing disk heads using ZBC instruction set.

**Trampoline** → Dropbox system that could dump data to S3 as elastic overflow when storage capacity ran close to the limit.

**FMEA (Failure Mode and Effects Analysis)** → Threat modeling process involving a large spreadsheet of every possible failure mode (including existential ones like data center fires), used to design protections at Dropbox.

**Convex** → James's company: a transactional database with TypeScript stored procedures, serializable isolation, automatic reactivity, and WebSocket subscriptions. Currently building lower-level primitives (fork, singletons, condition variables) as a "new operating system" for distributed applications.

**Jamie** → James's co-founder at Convex. "We really get along well in this respect. We can only put up with doing things we actually care about and believe in."

**Drew** → Dropbox founder. James spoke with Drew early on about the desire to migrate off S3.

**Mike Judge** → Creator of Silicon Valley, also created Beavis and Butt-Head and Office Space. Former software engineer at Lockheed.

**Silicon Valley (TV show)** → HBO series James consulted on for seasons 2+. Many of the cringiest moments came directly from real tech industry stories.

**De La Soul** → Possibly the band whose album release on Dropbox caused the memory-pressure/congestion collapse incident (James said "I can't remember" and "might have been" them).

**Leases (paper)** → An old academic paper that simply invented the idea of time-based locks. James cites this as the kind of "cool era of systems research" where a paper could just be about one novel idea.

**Patricia trie** / **Distributed hash table** → Examples of more sophisticated approaches that new hires from academia would suggest for the storage mapping problem at Dropbox — which James rejected in favor of a simple MySQL cluster.

**Kafka vs. RabbitMQ** → Example of the type of infrastructure choice that Convex wants to abstract away so developers don't have to think about it.

**Vandermonde matrix** → The custom encoding matrix Dropbox developed for its erasure coding scheme.

**Shingled Magnetic Recording (SMR)** → Experimental disk technology Dropbox was among the first to use at scale.

**ZBC (Zone Based Block Control)** → Instruction set for directly addressing disk heads, used in the Discotech project at Dropbox.

**Paxos, Raft, View Stamp Replication** → Three consensus algorithms all based on virtual synchrony — "they're all basically the same thing."

**Two-phase commit / two-phase locking** → The standard approach for distributed transaction coordination that Granola was designed to improve upon.

**Anthropic** → Referenced as an example of a company paying huge amounts for engineers, and as a company that still does whiteboard coding interviews despite being an AI lab. Also referenced for their "everyone's a member of technical staff" title structure that James views as "kind of a wink-wink thing."

**Google** → Referenced for "ruining systems research" by shifting the expectation toward industrial-scale deployed systems rather than novel ideas. Also referenced for people joining Dropbox from Google expecting infrastructure to already exist.

**AWS / Amazon S3** → Referenced extensively as the cloud provider Dropbox migrated from and still partners with. James calls them "a great partner."

**Lockheed** → Where Mike Judge started his career as a software engineer.

**Western Digital / Seagate** → Hard drive manufacturers Dropbox's supply chain team negotiated with directly.

**Office Space** → Mike Judge's film — "a dystopian cubicle era tech industry film."

**Go** → Language Dropbox rewrote the storage system in (while it was pre-GA). Great for concurrency and proxies but the garbage collector caused memory unpredictability issues for storage nodes.

**Rust** → Language the storage nodes were eventually rewritten in (while pre-GA) to eliminate memory unpredictability and enable the Discotech project.

**Python** → The language the original Magic Pocket prototype was built in. James defends it as "pretty efficient for I/O" but acknowledges it's not great for concurrency or refactoring.

**TypeScript** → The language Convex transactions are written in, running as stored procedures.

**PostgreSQL** → Referenced as a fine tool that "doesn't make problems go away" and whose "obvious way" of doing table scans doesn't scale for certain workloads.

**Claude** → Anthropic's AI model, referenced as an example of an AI coding tool. James says "Claude is not good at designing distributed systems protocols right now" and that asking it about new API design "is not going to give a good answer."

**ChatGPT** → Referenced as a tool that can give good answers about technical concepts but doesn't produce wisdom.

**Ralph (RALPH)** → An AI coding tool/agent that was trending on Twitter for a month and then disappeared — James's example of tech tabloidism/noise.

**Jensen [Huang]** → NVIDIA CEO, referenced as an example of tech tabloid celebrity whose statements people follow without it mattering for their actual engineering growth.

**Boris** → Possibly an AI executive/public figure, referenced alongside Jensen as someone whose public statements get treated as news but don't help engineers grow.

**Beyoncé** → Used in a metaphor: following AI news is "like reading about Beyoncé, but you're a nerd, and so you're reading about Jensen."

**Puppet / Chef** → Configuration management tools used as an example of system bias: a "Puppet team" defending their tool rather than solving the configuration management problem.

**996** → Chinese work culture (9am-9pm, 6 days/week), referenced as performative and "not real."