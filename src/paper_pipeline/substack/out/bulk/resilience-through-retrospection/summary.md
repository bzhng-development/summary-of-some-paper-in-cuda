<sup>Source: <https://www.developing.dev/p/resilience-through-retrospection></sup>

Here are the takeaways from this Ryan Peterman essay (a framework for incident retrospectives that reduces every review to three exhaustive questions):

**The universal incident timeline**

Ryan observes that every system breakage follows the same high-level timeline: users experience a degraded experience from the moment a problem starts until the moment it is mitigated. This structure means there are exactly three levers available to reduce user impact—detect faster, mitigate faster, or prevent the problem entirely. "To minimize impact there are only three options," and those three options directly motivate the three questions he asks in every retrospective. The framework isn't an abstraction layered on top of incidents; it falls directly out of the shape of the timeline itself.

**How can we detect this problem faster?**

"We can't fix problems we don't know about." The first question targets the gap between the start of an incident and the moment someone realizes something is wrong. In teams with less sophisticated observability, that gap can stretch for days until a user report finally surfaces the issue. Almost all detection discussions converge on automated alerting with a target of notification within 30 minutes of onset—which usually means adding logging where none exists and wiring alerts on top of it. He caveats that some signals prove genuinely hard to alert on because they lack stability, and "noisy alerts can be worse than having no alert at all." The principle is: an alert you learn to ignore trains the team *out* of noticing, so an undetectable problem is sometimes preferable to a false-alarm firehose.

**How can we mitigate this problem faster?**

Once the team knows there is an issue, the priority shifts to assembling the right people and diagnosing the root cause. Common tactics here include making debug logs richer and easier to analyze—turning the frantic grep-and-pray moment into something faster and more surgical. But diagnosis is only half of it; the team still has to deploy a fix. Ryan urges thinking through the deployment cycle of the specific platform that broke and asking whether there's a way to shorten it. The ideal case is a **feature flag already in production** that can redirect traffic within minutes. When that doesn't exist, the retrospective discussion naturally turns to "how to add something like that for the future"—treating the absence of a fast rollback or traffic-shifting mechanism as a vulnerability to close, not an immutable property of the system.

**How can we prevent this problem from happening again?**

This is Ryan's favorite question "because it is the most impactful." He sets a high bar: "I always aim to leave retrospective discussions such that, if we complete the agreed upon follow-ups, the incident can't happen again." The tactics fall into three ascending buckets. First, **catching bugs earlier**—introducing tests, static analysis, canary deployments, and type systems that catch the fault before it ever reaches users. Second, **building self-healing systems**—adding retries or backfill mechanisms so the system recovers without human intervention. Third, **making bugs impossible**—removing dependencies, refactoring code to eliminate the dangerous path, or adding fallbacks so the failure mode simply no longer exists. The range moves from "catch it sooner" to "survive it gracefully" to "remove the possibility entirely."

**The case for getting involved even if you didn't cause the incident**

Ryan closes with advice that broadens the audience beyond the person who broke production: get involved in your team's retrospective process regardless. Each discussion serves as a de facto system-design debrief, walking through the architecture and exactly why it failed. "Thinking through how to improve the system is a great exercise to build your technical skills." It's a form of learning grounded in real failure modes rather than idealized diagrams. The post itself is the product of "hundreds of retrospectives" he's run for his engineering org—a passion he traces back to "ever since I broke prod for the first time." The three-question framework isn't theoretical; it's battle-tested pattern recognition compressed into a checklist any team can adopt immediately.

---

The framework's power is its exhaustiveness: by mirroring the universal incident timeline, the three questions leave no lever unpulled. The **feature flag already in production** ideal threads through both mitigation and prevention—it is simultaneously the fastest mitigation tool and, when absent, the most concrete prevention action-item a retrospective can produce. Ryan's insistence that follow-ups should make recurrence *impossible* raises the bar from "less likely" to a design constraint, and his warning about noisy alerts captures a deeper tradeoff: resilience isn't just about adding mechanisms, but about ensuring those mechanisms preserve the team's attention rather than eroding it.
