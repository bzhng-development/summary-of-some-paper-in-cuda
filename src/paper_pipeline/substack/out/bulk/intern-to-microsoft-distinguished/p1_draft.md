Here are the takeaways from this David Fowler interview (went from intern to Distinguished Engineer at Microsoft across 11 promotions in 17+ years):

**Microsoft’s leveling system**

The Microsoft leveling system starts at 59 (entry-level, equivalent to SWE 1 elsewhere) and goes up to 80. There are level bands: 59-60 is SDE 1/junior, 61-62 is SDE 2, 63-64 is senior, and principal spans three levels. The title reflects the band, so you “can't tell from someone's title which notch they're at.” The principal band is especially broad; “the depth of experience of the people in that band is immense.” Beyond principal are the partner bands, then Technical Fellow and distinguished/senior fellow level. He never learned why it starts at 59, but suspects there are non-engineering levels below it.

**Early career and the Bill Gates era interview**

Fowler grew up in Barbados, left at 18 for Florida Tech, and originally hoped to do computer engineering until the first class convinced him he just wanted to code. At his first career fair, he and a friend differentiated themselves by “bringing a laptop showing our game demo to all the recruiters.” He got four internship offers from that. At Microsoft in 2006, the interview process was famously puzzle-based, inspired by *How Would You Move Mount Fuji?* In his on-campus interview, after a palindrome coding question done on paper, they asked him the *Die Hard 3* water-jug puzzle: “You have infinite water, and you want to make four gallons with a three and a five jug.” The last question was “how do you count the number of gas stations in Florida?” His final loop included “how do I add two numbers in base negative two?” He was among the last set of interns invited to Bill Gates's house, a surreal event where phones were confiscated, buses parked at a church, and he famously turned around to find Gates standing behind him and shook his hand.

**First promotion and the **code is currency** philosophy**

His first promotion came eight months in. Beyond assigned work, he trolled forums looking for feedback on Microsoft products and would “build features to solve them.” He showed the results to product managers, teammates, and architects, who “turned out were my sponsors.” His operating model was: **code is currency**. “I can show you a working sample. It's not PowerPoint, it's not a document. Here's a working example of something.” One director told him years later that he would tell people, “David has energy, go talk to him.”

**NuGet and the impact of a package manager**

NuGet, .NET’s package manager, started from competitive pressure. Microsoft was building a new web stack to compete with the LAMP stack, and Ruby on Rails/RubyGems had just emerged as a threat. In a meeting with Scott Guthrie (now an EVP), a PM mocked up a WordPress-style in-browser package manager, and Fowler became NuGet’s first engineer. The small founding team included architect David Ebbo and Phil Hack. Performance reviews around NuGet were glowing, but Fowler notes that the bigger promotions across bands required conglomeration of past success: “It was definitely not a one-off… It was about showing repeated impact over the last 10 years and whether we want to keep you going forward.”

**Bootstrapping SignalR and the **agency hack****

SignalR was a nights-and-weekends project born from watching Google Docs co-editing and wondering, “can we do this for code?” He taught himself about web sockets before they were released. A PM named Damon Edwards was building a talk on long-running streaming apps in .NET; Fowler saw it and said, “Hey, I'm building this thing. We should collaborate on this new library.” They open-sourced it, usage grew, and eventually a Microsoft PM said “we should make this official.” It became his first experience building a project from scratch entirely on his own initiative.

Ryan notes this was permissionless agency. Fowler’s rule: “you can just do things.” His pattern was to attach himself to someone whose career he wanted—in this case, architect David Ebbo—and mimic their behaviors. “I did this gradual thing where I attached myself to people who were like that. Then I found myself in that same position.” He’d build a prototype first, then show advocates. Early on he built “a thousand things, and maybe one would hit.” As his track record grew, the dynamic shifted: “the more success you have, the more leeway you get to fail, which gives you space to try more things.” High-trust engineers become “a weapon” that management says “we trust you, go do that thing you do.”

**Principal promotion and scaling through .NET Core**

When Fowler got senior, his boss warned him promotions would slow down. The jump to principal came through being pulled onto .NET Core: his director told him “we have this new thing spinning up and we need you to go work on it.” He and one other principal engineer were named architects of a complete cross-platform rewrite—free rein, 30 engineers, terrifying scope. Within a year he burned out trying to co-review every change.

The key shift was learning to delegate. On the smaller SignalR team, he had a moment where he realized the new hires “are gonna take longer at first, but once they understand all the stuff, then we can amplify the impact. Once I saw that, it flipped my brain.” He calls the leap to principal **outcome-focused leadership**: “being able to do more through others than yourself.” Getting there requires letting go of control-freak tendencies. “Software engineering is a collaborative event… It's difficult to just type harder and make a bigger impact.” He also had to learn not to treat everything as equally urgent and to let people fail.

**The architect archetype**

At Microsoft, architect is a semi-official role, available at middle principal bands, focused on breadth over depth. Fowler describes it as “vetting the decisions that are gonna last for 20 years.” He was the architect for .NET Core, overseeing the design of the whole stack—the web framework, web server, how they ship packages, the entire engineering idea. He wasn’t the engineering lead; he was the person asking, “If we do this change now, we can't do this thing in five or ten years.” He had mentors who had done it before and had been “watching and shadowing and trying to understand through experience how to build systems that last 20 years.”

**The path to Distinguished Engineer**

Distinguished Engineer (level 70, the final partner level) requires a formal peer-review process where a set of technical leaders decide if you’re worthy. Fowler provided his leadership with raw material—papers, code contributions, project histories—but didn’t drive the case himself. The evaluation leaned heavily on **deferred impact**: the email announcing his promotion cited .NET Core’s impact, work he’d done seven years prior. “It wasn't like I did it last year or the year before.” The process was surreal because peers at that level included Guido van Rossum (creator of Python). “Maximum imposter syndrome. I'm young for that role too.” After the decision, fellows messaged him saying “well deserved, and it's about time,” which helped.

**Mentorship and the **amplify your strengths** model**

Fowler was deliberate about mentors. When he became a partner, he told management he wanted a mentor who was a fellow because “if I wanted to be one, I should have mentors who can help me get there.” His primary mentor was Jeffrey Snover, inventor of PowerShell. Snover was direct where others sugarcoated: “Let's go to the source, talk to the CTOs, and figure out what it means for you in your role.” That meant asking the people who actually decide what distinguished-level work looks like in his specific context.

A different mentor helped with a 360-degree feedback package that identified his weaknesses. His impulse was to fix everything he was bad at. The mentor’s counter: **take your strengths and amplify them even more**. “You could work on the things you're not good at and be average at those, or you can work on things that you're really good at and be superhuman at those things… No one gets promoted to these roles for being average.” That reframed his entire approach: if he’s bad at scheduling and shipping logistics, he makes sure someone else owns that so he can focus on what he’s superhuman at.

**The meeting trap and guarding maker time**

Upon reaching partner, Fowler “fell steeply into the meeting trap of being in meetings all day, every day, and I got really sad.” His mentor’s rule: “as an IC, your job is to think and build stuff. So minimally, on your calendar, block eight hours a week to do that stuff.” He learned to say no to meetings aggressively. He still codes every day—his team sees lots of pull requests—and refuses to be “the architect who doesn't know the code and is giving people instructions.” He wants to be “everyone's peer” and face the builds. For his performance review, coding is no longer the measure; “I don't know if I get rewards anymore for coding. I get rewards for outcomes.”

**Engineers he looks up to**

His current team has five partner-level engineers, each remarkable in a different archetype: a super-coder who “produces way more code than everyone else combined,” an encyclopedic engineer who knows “obscure things you would never understand in your life,” and engineers with impeccable judgment. Anders Hejlsberg, creator of C#, is on his design review committee and models how to give feedback “in a way that is not talking down to you” despite having “invented four languages that everyone uses.” Mark Russinovich, CTO of Azure, will send emails that are “very dev-centric… I wrote some code, I improved the performance, and here's the result.” Fowler finds that deeply inspiring: “whenever I see high-level engineers still coding like that, I get really inspired.”

**Big-company tips: reorgs, interviewing, and the two skill sets**

On reorgs: “the only constant is change.” Before joining a new team, ask when the last reorg was. It tells you where the org is heading and what caused the shift. “Every new org is a whole new company.” When considering a role change, he advises mentees to focus not on what you love but on what you hate—the things you can’t stand—because those are hardest to work through.

On tech interviews: the industry over-indexes on writing new code. Fowler believes debugging is a completely separate, higher-leverage skill. He’d design an interview where “you give someone a crash dump… Figure out what the problem is.” Watching someone’s diagnostic process reveals far more. There’s also no university course that teaches **code archeology**: how to “surgically make your first change and not destroy everything.” Every assignment should be fixing one bug and writing a test in an artificially large, unfamiliar code base. “Computer science is not software engineering. It's not even close.”

Another insight: **the lower on the technology stack you sit, the fewer mistakes you're allowed to make**. Platform-level teams change something innocuous—like the order of return values—and “it just snaps some company's website down.”

**What kept him at Microsoft**

Fowler almost left when he was senior—his former boss recruited him to Twitter pre-IPO, and he also interviewed at GitHub, getting both offers. The EVP over-sold (“you're gonna be a distinguished engineer in five years”) and they countered, but money wasn’t the deciding factor. He stayed because of autonomy and network: “you build up a network of people, a track record. I could work on what I wanted to work on at that time.” He also engineered his career to avoid boredom: his LinkedIn shows “a new project every two years. That was by design.” He tells mentees in a situation where they can’t do their best work to leave, even if there’s a counteroffer. On Twitter’s IPO day, he admits, “instant depression, I was depressed for like a week.”

He also advocates working on one project long enough to feel the long-term consequences of your early decisions. “All the small issues you thought were small, that you could do later, will become big issues at some point… It gives you a whole new skillset and perspective on how to make decisions.”

**The Satya culture shift**

The culture changed “completely” from the Ballmer era. Early Microsoft had teams competing on the same project, building solutions “before knowing what the problem was,” and meetings where “if the most senior person was loud and aggressive, people who didn't think that way… would not get a chance to speak.” Post-Satya, collaboration became part of your review—you were dinged for duplicating something another team already built. Communication norms shifted: people raised hands on Teams, called on quiet participants, and “there was a lot more consideration for everyone else.” The famous XKCD comic of guns pointed at each other evolved from guns to ambivalence to collaboration.

**Regrets, work-life balance, and the building instinct**

His main regret is early-career arrogance and a lack of empathy: “Being right means nothing. It's just not important.” He had to learn that being right is “one of many things” and that tone in GitHub comments matters. He also regrets not negotiating his first salary—coming from Barbados, the offer was “the most money I've ever seen in my life,” and he didn’t know negotiation was possible. He tells every new grad to get multiple offers. (He also mentions, half-jokingly, not joining a friend’s Bitcoin company in 2008.)

On work-life balance: “there was no balance. There was just work hard, play hard.” A more recent reframe: balance is about what gives you energy versus what drains it. He works long hours because “programming for me was not just my job; it was my passion.” His GitHub is a “graveyard of projects”—fighting games, databases, distributed systems—each built just to learn. “Everything you build will teach you a new skill that you don't even know you have until later on.” People ask what book he read to get good, and his answer is always the same: “I wrote and read a lot of code.”