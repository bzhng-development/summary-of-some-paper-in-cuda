Here are the takeaways from this Ryan Peterman essay on shipping code faster (the fundamental skill that carries Junior to Mid-Level engineer):

**The lifecycle of a commit**

Ryan frames the entire optimization around understanding the five stages every code change passes through: design and planning, coding, testing, code review, and deployment. He was landing an average of five commits a day when he was promoted from Junior (IC3) to Mid-Level (IC4), and that pace came from intentionally debugging where his time was actually going at each stage. The distribution of time shifts dramatically with codebase familiarity—newcomers spend most of their time in design and coding, while veterans find testing and review becoming the bottlenecks.

**Optimizing when you're newer to the codebase**

For engineers still learning a system, the highest leverage comes from building mental infrastructure. "Understand the high-level flow"—know what the system does and where each request flows, and explicitly save code pointers to each entry point so you can trace your way through the code when needed. He also stresses learning editor navigation deeply: at a minimum, you should know how to search for arbitrary text and how to find function definitions ("usually cmd + click"). Pair programming gets a specific callout not just for getting unblocked, but for learning *how* to unblock yourself next time. "When they unblock you, don't just remember the answer. Think about how you can unblock yourself next time a similar issue comes up."

**Optimizing testing as you gain fluency**

Once the codebase is familiar, testing becomes the dominant cost. Ryan's playbook has four techniques. First, **use feature flags**—land all code changes behind gated code paths, then verify everything at once by turning the flag on for a small population and monitoring. This reduces risk, which in turn reduces the testing burden because "time spent testing your code should be proportional to how risky the change is." The bigger the cost when the code breaks, the more time you need to spend making sure it doesn't—and feature flags directly lower that risk. Second, **batch your testing**: stack multiple changes in a similar area and run tests covering the entire stack once, so "for a stack of N commits, you only need to pay the cost of testing once." Third, **test via automation**—if a quick test suite or script doesn't exist for a common code path, build it, because it'll save you and the team time repeatedly. Fourth, **reuse test plans**: use `git blame` to find old test plans and reuse them as a starting point for testing your own change.

**Getting through code review faster**

Code review is the one stage not fully in your control, but Ryan identifies three levers. **Write smaller diffs**—"the smaller your diffs are, the more willing people are to review them"; he'll even add `[easy]` to the title for his shortest changes to advertise this. **Reduce the number of iterations** by preempting any potential feedback or questions, aiming for approval on the first review. And **guide the reviewer** by explicitly pointing out which parts to scrutinize and providing relevant context through comments. If reviews aren't happening within a day of publishing, he suggests discussing it with the team—it's a team-level problem worth surfacing.

**The compounding approach**

The meta-lesson is incremental improvement: "Improve one thing each time you land code and you'll become much faster over time." The goal is reaching a point where you can land code changes within the same day you start working on them. This isn't about heroics; it's about systematically removing friction from each stage of the commit lifecycle, with the specific tactics changing as your experience in a codebase grows.

---

Ryan's entire playbook hinges on the deliberately unglamorous act of **debugging where you're spending the most time**—a systematic, stage-by-stage inspection of the **lifecycle of a commit**. The principle that **time spent testing your code should be proportional to how risky the change is** threads directly into the **feature flags** and **batch your testing** tactics, while the **reuse test plans** trick (via `git blame`) shows how much velocity comes not from raw speed but from not reinventing wheels. The code-review levers—**write smaller diffs**, **reduce the number of iterations**, **guide the reviewer**—are the social engineering complement to the technical testing optimizations, together forming a unified approach where every friction point in design, coding, testing, and review gets its own targeted countermeasure.