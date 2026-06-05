<sup>Source: <https://www.developing.dev/p/why-write-small-diffs></sup>

Here are the takeaways from this Ryan Peterman essay on engineering practice (a concise argument for small code changes distilled from writing over 1,000 diffs):

**The natural arc toward small diffs**

After writing over 1,000 diffs, Ryan found himself gravitating toward smaller ones. He notes that "often there’s wisdom in what you naturally tend towards over time"—this preference wasn't abstract dogma but something that "developed after lots of feedback and tweaks." His core conviction: "all engineers should be writing small diffs where possible." For those who haven't logged enough code to feel the benefits firsthand, he lays out five concrete reasons, each a practical argument reverse-engineered from experience rather than a style-guide rule.

**The bottleneck is review speed, not coding speed**

Once you know your codebase well, you realize "the bottleneck for landing code quickly isn’t in how fast you can write it. Often it’s in waiting for people to review it." Reviewers are simply more willing to pick up smaller diffs. Ryan uses a tactical label to signal this: he'll "add '[easy]' to the title for my shortest diffs to advertise this," lowering the psychological barrier for a busy teammate to jump in and keeping the review queue moving.

**Review thoroughness decays with size**

"The larger a change is, the more likely a reviewer is to skim the contents," letting low-quality code slip through undetected. Ryan wants real scrutiny—he frames a litmus test: "A true test of your code quality is if it invites feedback yet receives none." A small diff that no one nitpicks is genuinely clean; a large diff that draws no comments may simply have been too big to digest properly. Small diffs don't just get reviewed faster; they get reviewed *better*.

**Fewer bugs, fewer merge conflicts, cleaner rollbacks**

Small diffs are easier to reason about, so testing is more precise and regressions rarer: "The less a diff does, the easier it is to understand its effects." They also shrink the surface area for merge conflicts—"big changes have more surface area and take longer to write," increasing the odds of parallel edits colliding and wasting time on resolution. And when something does break, rollbacks stay simple; a large bad diff means "it’s more likely changes landed on top of it that also need to be reverted," turning a clean revert into a tangled operation.

**The single-things rule, with an escape hatch**

This doesn't mean large diffs are forbidden. "Sometimes it makes sense if you’re landing a bunch of boilerplate or auto-generated code." But the guiding principle is straightforward: "your diffs should aim to do just one thing." The essay is less a rigid policy and more a distillation of practical instinct—a pattern Ryan didn't adopt from a style guide but arrived at organically after a thousand iterations, then compressed into a compact argument for everyone else.

---

His **"[easy]" title tag** is a microcosm of the whole philosophy: make the reviewer's job so trivial they can't say no. The **"invites feedback yet receives none" litmus test** ties review depth directly to diff size—you can't trust silence on a large change. And the **"do just one thing" rule of thumb** isn't about banning large diffs but about keeping every diff's purpose singular enough that speed, thoroughness, bug surface, merge surface, and rollback simplicity all compound in your favor. These aren't separate tips; they're five facets of one shift in how you think about shipping code.
