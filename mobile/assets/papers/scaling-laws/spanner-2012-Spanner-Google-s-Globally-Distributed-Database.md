# Spanner: Google’s Globally-Distributed Database

**URL:** [https://pdos.csail.mit.edu/6.824/papers/spanner.pdf](https://pdos.csail.mit.edu/6.824/papers/spanner.pdf)

## 🎯 Pitch

Spanner achieves externally consistent transactions at global scale by exposing clock uncertainty as a first-class API, using atomic clocks and GPS to bound time errors to under 10 ms. This lets the system enforce real-time commit ordering across continents with only a ~5 ms penalty, breaking the long-held assumption that globally synchronous replication is prohibitively slow.

---

## 1. Executive Summary

Spanner introduces Google’s globally-distributed database, which combines semi-relational schematized tables, SQL querying, and general-purpose transactions with a foundational novel time API called **TrueTime** that exposes bounded clock uncertainty. The paper describes how TrueTime enables externally-consistent distributed transactions — the first system to provide this guarantee at global scale — by assigning globally-meaningful commit timestamps and using two core mechanisms: commit wait (delaying visibility until a timestamp is guaranteed past) and safe time (tracking the maximum timestamp at which a replica is up-to-date, enabling lock-free read-only transactions and non-blocking reads in the past). The evaluation demonstrates that commit wait adds roughly 5 ms of latency, that Paxos replication keeps write latency approximately constant (14.4 ms) across 1–5 replicas, and that automatic leader failover restores throughput within ~10 seconds after an unplanned datacenter failure, establishing that strongly consistent global transactions are feasible at production scale only when the underlying clock uncertainty — typically under 10 ms via GPS and atomic clocks — is explicitly bounded and exposed to the system.

## 2. Context and Motivation

### The Core Problem: Databases at Global Scale Inherit the Fundamental Tension Between Consistency and Latency

The distributed systems community has long understood that there is a fundamental tradeoff between consistency and performance. If you want your data to be perfectly consistent across all replicas, you must pay a coordination cost — every write must be propagated and acknowledged before it becomes visible. If you want high performance and low latency, you must relax consistency guarantees, accepting that different replicas may serve stale or conflicting data for some period of time. This tension becomes dramatically sharper when replicas are separated by wide-area network (WAN) latencies on the order of tens to hundreds of milliseconds. Within a single datacenter, consensus protocols like Paxos can achieve reasonable throughput and latency; across continents, the physics of light-speed propagation makes every message exchange painful.

Spanner confronts this tension head-on. The paper's central ambition is to build a database that is simultaneously: **(1) globally distributed** across many datacenters for availability and geographic locality, **(2) synchronously replicated** so that write acknowledgments imply durability across multiple failure domains, and **(3) externally consistent** — meaning that transactions appear to execute in a single global serial order that respects real-time ordering (if transaction T1 commits before T2 starts, T1's effects are visible to T2). Before Spanner, no production system achieved all three at global scale.

This is not merely an academic exercise. The paper's motivating application is **F1**, a rewrite of Google's advertising backend originally built on a manually sharded MySQL deployment. This backend stored tens of terabytes of data (Section 5.4), which is modest compared to many NoSQL systems but large enough that manual sharding became untenable. The MySQL sharding scheme tied each customer and all related data to a fixed shard based on business logic, and resharding as the customer base grew took **over two years of intense effort** involving coordination across dozens of teams. Beyond the operational nightmare, the MySQL deployment forced the team to compromise on both transactional correctness (some data had to be stored in external Bigtables, breaking cross-data queries) and availability (MySQL master-slave failover risked data loss and downtime).

The specific problem Spanner addresses is therefore: **can we build a database that provides serializable transactions, synchronous cross-datacenter replication, and a SQL interface, without sacrificing the scalability and availability that NoSQL systems achieved by relaxing these guarantees?** The answer, as the paper demonstrates, is yes — but only if you fundamentally rethink how time is represented in the system.

### The Clock Problem That Nobody Was Solving

To understand why this problem is hard, we must examine what "external consistency" actually requires. When a distributed transaction commits, the system must assign it a commit timestamp that: (1) reflects its serialization order relative to other transactions, and (2) respects real-time ordering — if human user A completes a purchase and then calls human user B to tell them about it, user B's subsequent query must see that purchase. These two requirements together mean the commit timestamp must be **greater than or equal to the absolute commit time** of all previously committed transactions, and **less than or equal to the time at which the transaction's effects become visible** to subsequent operations.

In a single-machine database, the system clock provides these timestamps trivially — you just read the clock at commit time. In a distributed database where replicas are separated by network delays, clocks on different machines inevitably disagree. The standard solution in the systems community was **loosely synchronized clocks** — run NTP, accept that clocks can be off by tens or hundreds of milliseconds, and build protocols that are correct despite clock skew (e.g., using logical clocks or vector clocks that track causal ordering rather than wall-clock time). This works for establishing relative ordering but gives you no leverage on real-time ordering. If two transactions commit on different continents, a loosely synchronized clock cannot tell you which happened first in absolute time, because the clock error bound exceeds the propagation delay of the causal signal.

Prior work had explored two directions, neither of which achieved the combination of guarantees that Spanner aims for:

### Prior Approach 1: Weaken the Consistency Guarantees (The NoSQL Path)

Systems like Bigtable and Dynamo achieved global scale by providing only **eventually consistent** replication. Bigtable supported cross-datacenter replication, but a write to one datacenter might not be visible at another for an unbounded duration. This is sufficient for many applications — analytics, content serving, offline processing — but breaks down for applications with transactional semantics. Users of Bigtable "consistently received complaints" (Section 1) about the difficulty of building applications requiring strong consistency, especially those with evolving schemas or wide-area replication requirements. The Percolator system was built explicitly to address Bigtable's lack of cross-row transactions, providing snapshot isolation over Bigtable — but Percolator was a single-datacenter system and did not attempt global synchronous replication.

The fundamental shortcoming of this approach is that **application correctness becomes the developer's responsibility**. If your application logic requires that a debit and credit happen atomically, or that a user profile update is immediately visible after completion, you must layer these guarantees on top of an eventually consistent store — a task that is notoriously error-prone and creates complex dependencies between application code and infrastructure behavior.

### Prior Approach 2: Strengthen Consistency Within a Single Datacenter (The Traditional Database Path)

Megastore, another internal Google system, took a step toward reconciling these tensions. It provided a semi-relational data model with synchronous replication across datacenters and a schema language similar to what Spanner would adopt. Over 300 Google applications used Megastore — including Gmail, Picasa, Calendar, and Android Market — despite its "relatively poor write throughput" (Section 1). The paper cites Megastore's popularity as direct evidence that application developers want schematized tables, synchronous replication, and strong consistency, even at significant performance cost.

But Megastore had two critical limitations:

1. **Poor write performance.** Megastore was layered on top of Bigtable, which imposed high communication costs. More fundamentally, Megastore did not support long-lived Paxos leaders — multiple replicas could initiate writes, and all writes from different replicas necessarily conflicted in the Paxos protocol, even if they did not logically conflict. The result was that **throughput collapsed on a Paxos group at several writes per second** (Section 6). This made Megastore unsuitable for high-throughput workloads.

2. **No external consistency.** Megastore provided synchronous replication but did not guarantee external consistency across the database. Two transactions completing on different Paxos groups could be assigned timestamps that did not respect real-time ordering, because Megastore lacked the timing infrastructure to enforce this.

The paper positions Megastore as a **proof of demand** rather than a successful solution. The fact that 300 applications chose to use a system with such severe write-throughput limitations demonstrates that the demand for strongly consistent, synchronously replicated storage with a relational data model is real and substantial. Spanner aims to deliver those same benefits without the performance penalties.

### Prior Approach 3: Reified Clock Uncertainty in Theory, But Not at Scale

The idea that bounded clock uncertainty could be used for concurrency control was not entirely novel. Adya et al. (1995) had explored efficient optimistic concurrency control using loosely synchronized clocks. Liskov (1993) had discussed practical uses of synchronized clocks in distributed systems more broadly. Farsite (Douceur and Howell, 2003) had derived bounds on clock uncertainty relative to a trusted reference and used them for server leases — conceptually similar to how Spanner uses Paxos leader leases.

However, all of these systems used **much looser clock bounds** than TrueTime achieves. Farsite's bounds were on the order of seconds or more. No prior system had demonstrated that clock uncertainty could be reduced to single-digit milliseconds at datacenter scale and then used as the **linchpin** of external consistency for a production database. The key insight is quantitative, not conceptual: if ϵ (half the uncertainty interval) is tens of milliseconds, then the commit-wait delay that external consistency requires — at least 2ϵ — becomes unacceptably large for interactive applications. TrueTime's achievement of ϵ < 10 ms (and typically 1–7 ms, with ϵ ≈ 4 ms most of the time, per Section 3) is what transforms clock uncertainty from a theoretical curiosity into a practical mechanism.

### Conflicting Industry Narratives

The paper is also motivated by a broader debate within the systems and database communities about whether general-purpose transactions are feasible or desirable at scale. Some prominent voices had argued against them:

- **"Life beyond Distributed Transactions" (Helland, 2007)** argued that distributed transactions are fundamentally unscalable and that applications should be designed to work without them.
- **Bigtable's design** explicitly avoided cross-row transactions, on the grounds that "general two-phase commit is too expensive to support, because of the performance or availability problems that it brings" (Section 2.3, citing various systems).
- **Stonebraker's critiques** of NoSQL (2010) argued that enterprises were uninterested in NoSQL precisely because of the missing transactional guarantees, but the NoSQL movement had largely accepted that strong consistency and global scale were incompatible.

Spanner takes a clear position in this debate: **"it is better to have application programmers deal with performance problems due to overuse of transactions as bottlenecks arise, rather than always coding around the lack of transactions"** (Section 2.3). The argument is pragmatic — developers can optimize hot spots when they appear, but retrofitting transactional semantics onto a system that lacks them is a pervasive, ongoing burden that distorts application architecture. Running two-phase commit over Paxos mitigates the availability problems that traditional two-phase commit introduces, because both the participant state and the coordinator state are replicated.

### How This Paper Positions Itself

The paper's contribution is not a single breakthrough but rather a **synthesis of ideas from two communities**:

- From the **database community**: a familiar semi-relational interface with schemas, SQL, and general-purpose transactions. The paper acknowledges the influence of Megastore's data model, the popularity of Dremel for SQL-based interactive analysis, and the consistent complaints about Bigtable's lack of cross-row transactions.
- From the **systems/distributed systems community**: scalability, automatic sharding, fault tolerance, consistent replication via Paxos, and wide-area distribution. The paper builds on Google's existing infrastructure (Bigtable's tablet model, Paxos for replication, Colossus for storage) and the growing body of work on Paxos-based state machine replication.

The **novel synthesis** is TrueTime. By reifying clock uncertainty as a first-class API rather than hiding it behind best-effort NTP synchronization, Spanner makes it possible to assign globally-meaningful commit timestamps without a centralized timestamp oracle. This, in turn, enables external consistency, lock-free read-only transactions (which execute at a system-chosen timestamp without blocking writers), non-blocking reads in the past (snapshot reads at any sufficiently up-to-date replica), and atomic schema changes across thousands of servers — all features that previously required tradeoffs between consistency and performance.

The paper's philosophical stance is captured in its concluding sentence: **"As a community, we should no longer depend on loosely synchronized clocks and weak time APIs in designing distributed algorithms"** (Section 8). This is a call to action — the existence of TrueTime demonstrates that bounded clock uncertainty is achievable at scale, and once it is achieved, a whole class of previously intractable distributed consistency problems becomes tractable.

### The Five-Year Iteration: Incremental Realization

An important contextual detail appears in the conclusions: Spanner's development took over five years, and part of that time was spent realizing that the system "should do more than tackle the problem of a globally-replicated namespace, and should also focus on database features that Bigtable was missing" (Section 8). This suggests that Spanner's initial charter may have been narrower — perhaps just a geographically replicated key-value store — and that the full vision of a globally-consistent relational database with SQL emerged incrementally as the team built TrueTime and recognized the breadth of what it enabled. The paper's structure reflects this evolution: the database features (schematized tables, SQL, transactions) are layered on top of the replication infrastructure, and TrueTime is the enabling technology that makes the layering coherent rather than a pile of incompatible abstractions.

In summary, the gap Spanner fills is the **absence of a production system that provides synchronous replication, external consistency, and a relational data model at global scale.** Prior work either achieved global scale by sacrificing consistency (Bigtable), achieved consistency at the cost of write throughput (Megastore), or achieved consistency within a single datacenter (traditional databases). Spanner synthesizes these threads by showing that a new primitive — bounded clock uncertainty exposed through a clean API — makes it possible to have all three simultaneously, and that the engineering required to achieve tight clock bounds (GPS, atomic clocks, Marzullo's algorithm for liar detection) is feasible at Google's operational scale.

</example>

## 3. Technical Approach

### 3.1 Reader Orientation

Spanner is a globally-distributed database that stores schematized, semi-relational data across hundreds of datacenters using synchronous Paxos replication, and it solves the fundamental problem of providing externally-consistent transactions at global scale by reifying clock uncertainty as a first-class API (TrueTime) and using it to assign globally-meaningful commit timestamps without a centralized timestamp oracle.

### 3.2 Big-Picture Architecture (Diagram in Words)

A Spanner deployment (called a **universe**) consists of the following components, organized hierarchically:

- **Zones**: The unit of administrative deployment and physical isolation — roughly analogous to an independent Bigtable deployment. Each zone contains one **zonemaster** and hundreds to thousands of **spanservers**. Zones are the unit across which data can be replicated, and they can be added or removed dynamically.

- **Zonemaster**: Assigns data to spanservers within its zone. It does not serve client requests directly.

- **Spanservers**: The workhorse servers that actually serve data to clients. Each spanserver manages 100–1000 **tablets**, each of which implements a single Paxos state machine storing a replicated bag of key-value mappings with timestamps. On top of the tablet, each spanserver implements a **lock table** (for two-phase locking concurrency control) and a **transaction manager** (for coordinating multi-Paxos-group two-phase commit), both only active at Paxos leaders.

- **Location Proxies**: Per-zone proxies that clients use to locate which spanservers serve their data. These are the discovery mechanism — clients ask the proxy, the proxy tells them which spanserver to talk to.

- **Universe Master**: A singleton console that displays zone-level status information for interactive debugging. It does not handle client traffic or data placement.

- **Placement Driver**: A singleton that handles automated data movement across zones on the timescale of minutes. It periodically communicates with spanservers to identify data that needs to be moved for load balancing or replication constraint satisfaction.

- **Clients**: Application processes that issue reads, writes, and transactions. Clients communicate directly with spanservers after locating them via proxies.

- **TrueTime Infrastructure**: A set of **time master** machines per datacenter (GPS-equipped and atomic-clock-equipped) and a **timeslave daemon** on every machine. The daemons poll multiple time masters, apply Marzullo's algorithm to detect and reject liars, and compute a continuously updated uncertainty interval `[earliest, latest]` that is guaranteed to contain absolute time. This interval is exposed to Spanner through the TrueTime API.

Information flows as follows: a client consults the location proxy → the proxy directs it to the appropriate spanserver → the spanserver's Paxos leader processes the request using the lock table and transaction manager → commit timestamps are assigned using TrueTime's `TT.now()` → replication occurs via Paxos to other replicas in the group → the placement driver periodically rebalances data in the background by moving directories between Paxos groups.

### 3.3 Roadmap for the Deep Dive

- **First**, the spanserver software stack and tablet structure — this is the lowest-level data storage and replication primitive upon which everything else is built, so understanding it is prerequisite.
- **Second**, directories and placement — how data is bucketed for replication control and locality, and the `movedir` mechanism for migrating data between groups without blocking operations.
- **Third**, the data model — how schematized semi-relational tables with `INTERLEAVE IN` are layered on top of the key-value store, because this defines the interface applications see and the locality relationships they can express.
- **Fourth**, the TrueTime API and implementation — the enabling technology for external consistency, described before concurrency control because every timestamp assignment decision depends on it.
- **Fifth**, Paxos leader lease management using TrueTime — how the disjointness invariant is enforced without synchronous lease logging, because leader leases are the foundation for monotonic timestamp assignment.
- **Sixth**, timestamp assignment for read-write transactions — the start and commit-wait rules, the two-phase commit protocol, and the formal proof of external consistency.
- **Seventh**, timestamp assignment for read-only transactions and snapshot reads — the safe time mechanism (`tsafe`), how it enables lock-free consistent reads at any sufficiently up-to-date replica, and refinements to avoid false conflicts.
- **Eighth**, atomic schema-change transactions — how TrueTime enables non-blocking schema updates across thousands of servers by assigning a future timestamp and synchronizing operations against it.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems design paper** whose core idea is that exposing bounded clock uncertainty through a clean API (TrueTime) makes it possible to assign globally-meaningful commit timestamps in a distributed system, which in turn enables external consistency, lock-free read-only transactions, and non-blocking schema changes at global scale. The paper describes the full architecture of Spanner as the embodiment of this idea.

---

#### Spanserver Software Stack and Tablet Structure

Each spanserver manages between 100 and 1000 instances of a data structure called a **tablet**. A tablet implements a bag of the following mappings:

$$(\text{key} : \text{string}, \text{timestamp} : \text{int64}) \rightarrow \text{string}$$

where `key` is a string identifying the data row, `timestamp` is an int64 timestamp assigned by Spanner, and the mapping produces a string value. This is a temporally versioned key-value store: unlike Bigtable, Spanner assigns timestamps to data, making it a multi-version database where every write creates a new version tagged with its commit timestamp rather than overwriting the previous value.

**What it computes:** for a given database row (identified by key) and a given timestamp, the tablet returns the value that was committed as of that timestamp. Multiple versions of the same key can coexist at different timestamps.

**Why this form:** the addition of timestamps to the key-value mapping is what transforms Spanner from a plain key-value store into a temporal database. It enables reads in the past (snapshot reads at any timestamp), versioned garbage collection (old versions can be purged according to configurable policies), and a consistent global ordering of all writes (since commit timestamps are globally meaningful and monotonically increasing). Without timestamps, every read would see only the most recent value, and there would be no way to perform a consistent snapshot across multiple keys.

A tablet's state is stored in a set of B-tree-like files and a write-ahead log, all on **Colossus**, Google's distributed file system (the successor to GFS). This is the same storage architecture as Bigtable's tablet.

To support replication, each spanserver implements **a single Paxos state machine on top of each tablet**. This is a deliberate simplification from an earlier Spanner incarnation that supported multiple Paxos state machines per tablet for more flexible replication configurations — the complexity of that design led the team to abandon it. Each Paxos state machine stores its metadata and log in the corresponding tablet's Colossus files. The set of replicas for a given tablet is collectively called a **Paxos group**.

The Paxos implementation supports **long-lived leaders** with time-based leader leases, whose length defaults to 10 seconds. This is a critical design choice: having a long-lived leader means that for the vast majority of time, all writes to a Paxos group are initiated by a single replica, which enables efficient lock management (the lock table lives at the leader) and low-latency reads (reads can go to the leader without additional Paxos rounds). The implementation logs every Paxos write twice — once in the tablet's log and once in the Paxos log — a choice the paper describes as having been "made out of expediency" and "likely to remedy eventually." The Paxos implementation is pipelined to improve throughput in the presence of WAN latencies, but writes are applied in order — this ordered application is a property that Section 4's timestamp management depends on.

At every replica that is a leader, each spanserver implements a **lock table** for concurrency control. The lock table contains the state for two-phase locking: it maps ranges of keys to lock states (indicating which transactions hold read or write locks on which key ranges). Operations that require synchronization — transactional reads and writes — acquire locks in the lock table; other operations bypass it. The lock table exists only at leaders because locks are managed by a single authority per Paxos group, and the long-lived leader ensures that this single authority is stable.

At every replica that is a leader, each spanserver also implements a **transaction manager** to support distributed transactions that span multiple Paxos groups. The transaction manager implements the participant leader role in two-phase commit; the other replicas in the group are participant slaves. If a transaction involves only a single Paxos group — which the paper notes is the case for most transactions — it can bypass the transaction manager entirely, since the lock table and Paxos together provide atomicity and isolation within the group. For multi-group transactions, one participant group is chosen as the **coordinator**: its leader becomes the **coordinator leader**, and its slaves become **coordinator slaves**. The state of each transaction manager is stored in the underlying Paxos group and is therefore replicated — this is what makes the two-phase commit protocol fault-tolerant even when the coordinator fails.

---

#### Directories and Placement

On top of the key-value mappings, Spanner implements a bucketing abstraction called a **directory**, which is a set of contiguous keys that share a common prefix. The paper notes that the term "directory" is a historical accident and that "bucket" would have been a better name. The common prefix arises from Spanner's data model: interleaved tables create keys where a parent row's primary key is a prefix of its child rows' keys (explained in Section 3.4, Data Model).

A directory is the **unit of data placement**. All data in a directory has the same replication configuration — the same number of replicas, the same geographic placement constraints, the same types of replicas. When data is moved between Paxos groups, it is moved directory by directory. Spanner might move a directory to:
- Shed load from a Paxos group whose write throughput has become a bottleneck.
- Co-locate directories that are frequently accessed together into the same Paxos group (so that transactions spanning those directories become single-group transactions).
- Move a directory into a Paxos group whose replicas are geographically closer to the clients that access it most frequently.

Directories can be moved while client operations are ongoing — the data migration happens in the background without blocking reads or writes on the directory. The paper states that a 50 MB directory can typically be moved in a few seconds.

An important structural consequence follows: a Paxos group may contain multiple directories, implying that a Spanner tablet is **not necessarily a single lexicographically contiguous partition of the row space** (unlike a Bigtable tablet, which is exactly one contiguous range). Instead, a Spanner tablet is a container that may encapsulate multiple non-contiguous row-space partitions. The design choice is intentional: by allowing a tablet to hold multiple directories with different key prefixes, Spanner can co-locate directories that are frequently accessed together (e.g., a user's profile data and their photos metadata) even if they would be far apart in key order. This sacrifices the simplicity of range-based partitioning for better locality and reduced cross-group transaction frequency.

**Movedir** is the background task that moves directories between Paxos groups. It is also used to add or remove replicas to Paxos groups, because Spanner (at the time of the paper) did not support in-Paxos configuration changes. Crucially, `movedir` is not implemented as a single transaction — if it were, the transaction would hold locks on the entire directory for the duration of the data copy (potentially seconds), blocking all concurrent reads and writes. Instead, the protocol works as follows:
1. `movedir` registers the fact that it is starting to move the directory, creating metadata that redirects future operations to the destination group.
2. It copies the data in the background, allowing reads and writes to proceed on the source group during the copy.
3. When it has moved all but a nominal amount of data, it uses a **single atomic transaction** to move the remaining nominal data and update the metadata for both Paxos groups, completing the handoff.

This two-phase approach (bulk background copy + atomic tail flip) minimizes the window during which the directory is unavailable.

A directory is also the smallest unit whose **geographic-replication properties** (placement) can be specified by an application. The placement specification language separates responsibilities:
- **Administrators** define a menu of named placement options along two dimensions: the number and types of replicas, and the geographic placement of those replicas. Example: "North America, replicated 5 ways with 1 witness."
- **Applications** tag each database or individual directory with a combination of those named options. Example: an application might store each end-user's data in its own directory, enabling user A's data to be replicated 3 ways in Europe, and user B's data to be replicated 5 ways in North America.

The paper notes that for expository clarity, it oversimplifies: Spanner will shard a directory into multiple **fragments** if it grows too large. Fragments may be served from different Paxos groups and therefore different servers. `movedir` actually moves fragments rather than whole directories. This means the unit of data placement is ultimately the fragment, not the directory, but applications specify placement at directory granularity and Spanner handles the internal fragmentation transparently.

---

#### Data Model

Spanner exposes a data model to applications that is based on **schematized semi-relational tables**, with a SQL-based query language and general-purpose transactions. This design was driven by the observed demand for Megastore's semi-relational model (300+ applications used Megastore despite its poor performance), the popularity of Dremel for SQL-based interactive analysis, and persistent complaints about Bigtable's lack of cross-row transactions. The paper takes a strong stance: "it is better to have application programmers deal with performance problems due to overuse of transactions as bottlenecks arise, rather than always coding around the lack of transactions."

The application data model is layered on top of the directory-bucketed key-value mappings. An application creates one or more **databases** in a universe. Each database can contain an unlimited number of schematized tables that look like relational-database tables (rows, columns, versioned values), with the constraint that every table must have an ordered set of one or more **primary-key columns**. This requirement is the residual key-value store heritage: the primary keys form the name for a row, and each table defines a mapping from the primary-key columns to the non-primary-key columns. A row exists only if some value (even `NULL`) is defined for its keys — there is no notion of an empty row with no values.

The primary-key requirement serves a crucial purpose: it lets applications control data locality through their choice of keys, because Spanner uses key prefixes to determine directory grouping.

The schema language includes `INTERLEAVE IN` declarations, which define **hierarchies of tables**. A client declares that one table's rows are interleaved (stored physically adjacent) with a parent table's rows. The table at the top of the hierarchy is called a **directory table**. The rule is: each row in a directory table with key `K`, together with all rows in descendant tables whose keys start with `K` in lexicographic order, forms a directory. This means a directory physically groups a parent row and all its related child rows together on the same Paxos group, enabling single-group transactions for operations that read or modify a parent and its children.

The `ON DELETE CASCADE` declaration specifies that deleting a row in the directory table automatically deletes any associated child rows — this is the standard referential integrity cascade from relational databases, but implemented within the storage layer because the parent-child grouping is physically co-located.

Figure 4 provides a concrete example: a `Users` table (primary key `uid`) is declared as a directory table, and an `Albums` table (composite primary key `uid, aid`) is declared as interleaved in `Users`. This means all albums for user `uid=2` are stored physically adjacent to the `Users(2)` row, and a transaction that reads both the user profile and their albums can execute as a single-group transaction. This interleaving of tables to form directories is significant because it lets clients describe locality relationships between multiple tables — without it, Spanner would not know the most important locality relationships that determine which data should be co-located for performance.

---

#### TrueTime API and Implementation

TrueTime is the novel time API that is the linchpin of Spanner's external consistency guarantees. The API is deliberately minimal, exposing clock uncertainty rather than hiding it:

| Method | Returns |
|--------|---------|
| `TT.now()` | `TTinterval: [earliest, latest]` |
| `TT.after(t)` | `true` if `t` has definitely passed |
| `TT.before(t)` | `true` if `t` has definitely not arrived |

`TT.now()` returns a `TTinterval` whose endpoints are of type `TTstamp`, with the guarantee that the interval contains the absolute time at which `TT.now()` was invoked. `TT.after(t)` and `TT.before(t)` are convenience wrappers that query whether timestamp `t` is definitely in the past or definitely in the future respectively, given the current uncertainty interval.

Define the **instantaneous error bound** as **ϵ**, which is half of the interval's width, and the **average error bound** as $\bar{\epsilon}$. The formal guarantee is:

$$\text{tt.earliest} \leq t_{\text{abs}}(e_{\text{now}}) \leq \text{tt.latest}$$

where `$t_{\text{abs}}(e)$` is the absolute time of event `$e$`, `$e_{\text{now}}$` is the invocation event of `TT.now()`, and `tt.earliest` and `tt.latest` are the endpoints returned by the call.

**What this equation states:** when you call `TT.now()`, the returned interval `[earliest, latest]` is guaranteed to bracket the true absolute time at the moment of invocation. The width of the interval is `$2\epsilon$`, where `$\epsilon$` is the worst-case clock error at that instant. The guarantee is absolute: the true time is never outside the interval.

**Why this form matters:** standard clock APIs (like `gettimeofday()`) return a single timestamp with no error bound, making it impossible to reason about whether an action happened before or after another action on a different machine. By explicitly representing uncertainty, TrueTime makes it possible to write protocols that are correct *despite* clock error — you simply wait until the uncertainty interval has passed before making a decision that depends on accurate time ordering. The cost of this waiting is proportional to `$\epsilon$`, so reducing `$\epsilon$` directly reduces the performance penalty of time-dependent correctness.

**Implementation.** TrueTime uses two forms of time reference — GPS and atomic clocks — because they have different, uncorrelated failure modes:
- **GPS vulnerabilities:** antenna and receiver failures, local radio interference, correlated failures (design faults like incorrect leap-second handling, spoofing), and GPS system outages. These can affect multiple GPS receivers simultaneously.
- **Atomic clock vulnerabilities:** they can fail in ways uncorrelated to GPS and to each other, and over long periods they drift significantly due to frequency error. But they do not share GPS's common-mode failures.

The infrastructure consists of **time master** machines per datacenter and a **timeslave daemon** on every machine. Most masters have GPS receivers with dedicated antennas physically separated to reduce the impact of antenna failures, radio interference, and spoofing. The remaining masters — called **Armageddon masters** — are equipped with atomic clocks. The paper notes that an atomic clock is not that expensive: the cost of an Armageddon master is of the same order as a GPS master.

All masters' time references are regularly compared against each other. Each master also cross-checks the rate at which its reference advances time against its own local clock, and evicts itself from the set of trusted masters if there is substantial divergence — this prevents a faulty master from corrupting the time synchronization of the entire datacenter. Between synchronizations, **Armageddon masters advertise a slowly increasing time uncertainty** derived from conservatively applied worst-case clock drift (this accounts for the physical fact that atomic clocks drift, and the longer it has been since the last cross-check, the larger the potential accumulated error). **GPS masters advertise uncertainty that is typically close to zero** because GPS provides continuous external calibration.

Every timeslave daemon polls a variety of masters to reduce vulnerability to errors from any single master. Some are GPS masters chosen from nearby datacenters (for low communication delay), while others are GPS masters from farther datacenters and some Armageddon masters (for diversity against correlated failures). The daemon applies a variant of **Marzullo's algorithm** to detect and reject liars — this is a classic distributed-systems algorithm that finds the smallest interval consistent with a majority of reported time intervals, effectively discarding outliers. The daemon then synchronizes the local machine clock to the non-liars.

To protect against broken local clocks, machines that exhibit frequency excursions larger than the worst-case bound derived from component specifications and operating environment are evicted — meaning they are taken out of service until the clock can be repaired or replaced, since a broken local clock could drift faster than the assumed worst-case rate, invalidating the uncertainty bounds.

Between synchronizations, a daemon advertises a slowly increasing time uncertainty. The error bound `$\epsilon$` is derived from:
1. Conservative worst-case local clock drift: 200 microseconds/second (the applied drift rate).
2. Time-master uncertainty (the error advertised by the trusted masters themselves).
3. Communication delay to the time masters (network latency introduces uncertainty about when exactly a reported time was valid).

In the production environment, `$\epsilon$` is typically a sawtooth function of time, varying from about **1 to 7 ms** over each poll interval. The sawtooth shape arises because immediately after a poll, uncertainty is low (dominated by communication delay ~1 ms), and it then linearly increases at 200 µs/s until the next poll resets it. The poll interval is 30 seconds, so the accumulated drift over one interval is 30 s × 200 µs/s = 6 ms, which, added to the 1 ms communication delay, gives the 1–7 ms range. The **average error bound $\bar{\epsilon}$ is therefore 4 ms most of the time.**

Excursions from this sawtooth are possible during failures: time-master unavailability can cause datacenter-wide increases in `$\epsilon$` (since daemons have fewer masters to poll), and overloaded machines or network links can cause localized `$\epsilon$` spikes (since communication delay increases).

The absolute time epoch is analogous to UNIX time, but with **leap-second smearing** — meaning leap seconds are handled by gradually adjusting the clock rate over a period rather than by inserting a discontinuous jump, which would break the assumption of bounded clock drift.

---

#### Paxos Leader Lease Management Using TrueTime

Spanner's Paxos implementation uses timed leases to make leadership long-lived (default 10 seconds). The lease mechanism works as follows: a potential leader sends requests for timed lease votes to replicas; upon receiving a quorum of lease votes, the leader knows it has a lease for a bounded interval. A replica extends its lease vote implicitly on every successful write (since participating in a write acknowledges that the leader is still alive), and the leader explicitly requests lease-vote extensions if they are near expiration.

Spanner depends on a critical **disjointness invariant**: for each Paxos group, each Paxos leader's lease interval is disjoint from every other leader's. In other words, at no point in absolute time can two replicas simultaneously believe they are the leader of the same Paxos group. This invariant is what prevents split-brain scenarios where two leaders assign conflicting timestamps or commit conflicting writes.

The simplest way to ensure disjointness would be for a leader to issue a synchronous Paxos write of its lease interval whenever it is extended, and for a subsequent leader to read that interval and wait until it has definitely expired before assuming leadership. This approach works but adds extra synchronous writes — one per lease extension — which are costly over WAN latencies.

TrueTime enables a more efficient approach that avoids these extra log writes. Define:
- `$v^{\text{leader}}_{i,r} = \text{TT.now().earliest}$` — the lower bound on the start of a lease vote from replica `$r$` to the `$i$`th potential leader, computed before the lease request is sent (`$e^{\text{send}}_{i,r}$`).
- Each replica `$r$` grants a lease at event `$e^{\text{grant}}_{i,r}$`, which occurs after receiving the request (`$e^{\text{receive}}_{i,r}$`). The lease ends at `$t^{\text{end}}_{i,r} = \text{TT.now().latest} + 10$`, computed after `$e^{\text{receive}}_{i,r}$`.
- A replica obeys the **single-vote rule**: it will not grant another lease vote until `$\text{TT.after}(t^{\text{end}}_{i,r})$` is true — that is, until the previous lease has definitely expired.
- When the `$i$`th leader receives a quorum of votes (event `$e^{\text{quorum}}_i$`), it computes its lease interval as:

$$\text{lease}_i = [\text{TT.now().latest}, \min_r(v^{\text{leader}}_{i,r}) + 10]$$

where `$\min_r(v^{\text{leader}}_{i,r})$` is the earliest start time across all replicas in the quorum, and the lease ends 10 seconds after that earliest start.

**What this computes:** the leader's effective lease interval — the window during which it may safely assign timestamps and serve as the leader without violating disjointness. The start is `TT.now().latest` (the leader's current upper bound on absolute time when it received the quorum). The end is 10 seconds after the *earliest* vote start time in the quorum — this is conservative because it uses the earliest vote, ensuring that even the replica that voted earliest will consider its vote expired before the leader's lease expires.

**Why this form works without synchronous logging:** the disjointness proof relies on the fact that consecutive leaders must share at least one replica in their quorums (by the definition of Paxos quorums). Call that replica `$r_0$`. The `$i$`th leader's lease ends at `$\min_r(v^{\text{leader}}_{i,r}) + 10 \leq v^{\text{leader}}_{i,r_0} + 10$`. By definition, `$v^{\text{leader}}_{i,r_0}$` was computed before the vote request was sent, which happened before `$r_0$` received it, which means `$v^{\text{leader}}_{i,r_0} + 10 \leq t^{\text{end}}_{i,r_0}$`. The single-vote rule ensures that `$r_0$` will not grant a vote to the `$(i+1)$`th leader until `$\text{TT.after}(t^{\text{end}}_{i,r_0})$`, which means the `$(i+1)$`th leader's grant event `$e^{\text{grant}}_{i+1,r_0}$` is strictly after `$t^{\text{end}}_{i,r_0}$`. By causality, `$e^{\text{grant}}_{i+1,r_0}$` happens before `$e^{\text{quorum}}_{i+1}$`, and by definition `$e^{\text{quorum}}_{i+1} \leq \text{lease}_{i+1}.\text{start}$`. The chain of inequalities proves `$\text{lease}_i.\text{end} < \text{lease}_{i+1}.\text{start}$` — the intervals are disjoint.

To enforce the single-vote rule across different incarnations of a replica `$r$`, Spanner logs a lease vote at the granting replica before granting it — but this log write can be piggybacked on existing Paxos-protocol log writes, so it does not add additional synchronous delays.

The paper also defines **abdication**: a Paxos leader can release its slaves from their lease votes. Before abdicating, the leader must wait until `$\text{TT.after}(s_{\max})$` is true, where `$s_{\max}$` is the maximum timestamp assigned by that leader. This preserves the disjointness invariant because no new leader will be elected until the old leader's lease has expired, and the old leader waits until any timestamps it assigned are definitely in the past before releasing its hold on leadership.

---

#### Timestamp Assignment for Read-Write Transactions

Transactional reads and writes use two-phase locking for concurrency control, which means they can be assigned timestamps at any time after all locks have been acquired but before any locks have been released. For a given transaction, Spanner assigns it the timestamp that Paxos assigns to the Paxos write representing the transaction commit — meaning the timestamp is embedded in the Paxos log entry that records the commit decision.

Spanner depends on a **monotonicity invariant**: within each Paxos group, Spanner assigns timestamps to Paxos writes in monotonically increasing order, even across leaders. A single leader can trivially enforce this by remembering the last timestamp it assigned and ensuring the next is strictly greater. Across leaders, the invariant is enforced using the disjointness invariant: a leader must only assign timestamps within the interval of its leader lease (which is disjoint from all other leaders' lease intervals), and whenever a timestamp `$s$` is assigned, `$s_{\max}$` is advanced to `$s$` to ensure that any subsequent leader (whose lease starts after this one ends) cannot assign a timestamp at or below `$s$`.

Spanner also enforces an **external-consistency invariant**: if the start of a transaction `$T_2$` occurs after the commit of a transaction `$T_1$`, then the commit timestamp of `$T_2$` must be greater than the commit timestamp of `$T_1$`. Formally:

$$t_{\text{abs}}(e^{\text{commit}}_1) < t_{\text{abs}}(e^{\text{start}}_2) \Rightarrow s_1 < s_2$$

where `$t_{\text{abs}}(e)$` is the absolute time of event `$e$`, `$e^{\text{commit}}_1$` is the commit event of `$T_1$`, `$e^{\text{start}}_2$` is the start event of `$T_2$`, and `$s_1$` and `$s_2$` are the commit timestamps assigned to `$T_1$` and `$T_2$` respectively.

**What this states:** if everyone in the world can agree that `$T_1$` finished before `$T_2$` began (in absolute, wall-clock time), then the database's assigned serialization order must place `$T_1$` before `$T_2$`. This is stricter than serializability alone: serializability only requires that there exists *some* total order consistent with the transactions' reads and writes, but external consistency additionally requires that the chosen order respects real-time precedence.

The protocol obeys two rules that together guarantee this invariant. Define the arrival event of the commit request at the coordinator leader for a write `$T_i$` as `$e^{\text{server}}_i$`.

**Start Rule:** The coordinator leader for a write `$T_i$` assigns a commit timestamp `$s_i$` **no less than** the value of `$\text{TT.now().latest}$`, computed after `$e^{\text{server}}_i$`. In operational terms: when the coordinator receives the commit request, it calls `TT.now()`, takes the `latest` endpoint, and ensures the assigned timestamp is `$\geq$` that value.

**Commit Wait Rule:** The coordinator leader ensures that clients cannot see any data committed by `$T_i$` until `$\text{TT.after}(s_i)$` is true. In operational terms: after assigning the timestamp `$s_i$`, the coordinator delays applying the commit record (making the transaction's effects visible to reads) until TrueTime guarantees that `$s_i$` is definitely in the past.

**Formal proof of external consistency** (reproduced from the paper to show the logical chain):

$$s_1 < t_{\text{abs}}(e^{\text{commit}}_1) \quad \text{(commit wait ensures } s_1 < \text{absolute commit time of } T_1 \text{)}$$

$$t_{\text{abs}}(e^{\text{commit}}_1) < t_{\text{abs}}(e^{\text{start}}_2) \quad \text{(by assumption: } T_2 \text{ started after } T_1 \text{ committed)}$$

$$t_{\text{abs}}(e^{\text{start}}_2) \leq t_{\text{abs}}(e^{\text{server}}_2) \quad \text{(causality: the start happens before the commit request arrives at the coordinator)}$$

$$t_{\text{abs}}(e^{\text{server}}_2) \leq s_2 \quad \text{(start rule: } s_2 \geq \text{TT.now().latest at commit request arrival)}$$

$$s_1 < s_2 \quad \text{(by transitivity)}$$

**Why these two rules together are sufficient:** the Start Rule ensures that `$s_2$` is assigned a timestamp at least as large as the absolute time when the coordinator received `$T_2$`'s commit request, which is at least as large as when `$T_2$` started. The Commit Wait Rule ensures that `$s_1$` is strictly less than the absolute time when `$T_1$`'s effects become visible, which is when `$T_1$` committed. If `$T_1$` committed before `$T_2$` started, then `$T_1$`'s commit time is before `$T_2$`'s start time, which is before `$T_2$`'s commit request reaches the coordinator, which is at or before `$s_2$`. Therefore `$s_1 < s_2$`.

**Practical consequence of commit wait:** the coordinator leader chose `$s$` based on `TT.now().latest`, and then waits until that timestamp is guaranteed to be in the past. The expected wait is at least `$2\bar{\epsilon}$` (since `$\epsilon$` is half the uncertainty, and the leader needs the upper bound to become a lower bound), which is approximately 8 ms given `$\bar{\epsilon} = 4$` ms. The paper notes that this wait is typically overlapped with Paxos communication (the coordinator is waiting for prepare acknowledgments and replicating the commit record during this window), so it is often not additional latency at the tail.

**Two-phase commit protocol details.** When a client has completed all reads and buffered all writes, it begins two-phase commit. The client chooses a coordinator group and sends a commit message to each participant's leader, containing the coordinator's identity and any buffered writes. Having the client drive two-phase commit avoids sending data twice across wide-area links — the client sends writes directly to each participant rather than funneling everything through the coordinator.

A non-coordinator-participant leader first acquires write locks. It then chooses a **prepare timestamp** that must be larger than any timestamps it has assigned to previous transactions (to preserve monotonicity), and logs a prepare record through Paxos. Each participant then notifies the coordinator of its prepare timestamp.

The coordinator leader also first acquires write locks but skips the prepare phase (because the coordinator's own locks and timestamps are part of the commit decision). It chooses a commit timestamp `$s$` for the entire transaction after hearing from all other participant leaders. The chosen `$s$` must satisfy all of:
- `$s \geq$` all prepare timestamps reported by participants (so that no participant has already committed a later transaction at a timestamp that should be ordered after this one — see the safe time discussion below).
- `$s \geq \text{TT.now().latest}$` at the time the coordinator received its commit message (the Start Rule).
- `$s >$` any timestamps the coordinator leader has assigned to previous transactions (monotonicity).

The coordinator leader then logs a commit record through Paxos (or an abort if it timed out waiting for participants). Before allowing any coordinator replica to apply the commit record and release locks, the coordinator leader obeys the Commit Wait Rule — waiting until `$\text{TT.after}(s)$` is true. After commit wait, the coordinator sends the commit timestamp to the client and all other participant leaders. Each participant leader logs the transaction's outcome through Paxos. All participants apply at the same timestamp and then release locks.

---

#### Safe Time: Serving Reads at a Timestamp

The monotonicity invariant enables Spanner to determine whether a replica's state is sufficiently up-to-date to satisfy a read at a given timestamp. Every replica tracks a value called **safe time** `$t_{\text{safe}}$`, which is the maximum timestamp at which the replica is up-to-date — meaning the replica is guaranteed to have seen the effects of all transactions that committed at or below that timestamp. A replica can satisfy a read at timestamp `$t$` if and only if:

$$t \leq t_{\text{safe}}$$

where `$t$` is the desired read timestamp and `$t_{\text{safe}}$` is the replica's current safe time.

**What this condition ensures:** if a read is allowed at timestamp `$t$`, the replica's state reflects every write that committed at or before `$t$`, and no future write will be applied at or below `$t$` that would change the state the replica has already applied. This is the guarantee required for snapshot isolation: every read at timestamp `$t$` sees a consistent snapshot of the database as of `$t$`.

Safe time is defined as:

$$t_{\text{safe}} = \min(t^{\text{Paxos}}_{\text{safe}}, t^{\text{TM}}_{\text{safe}})$$

where `$t^{\text{Paxos}}_{\text{safe}}$` is the timestamp of the highest-applied Paxos write at this replica, and `$t^{\text{TM}}_{\text{safe}}$` is the safe time derived from the transaction manager's state.

**`$t^{\text{Paxos}}_{\text{safe}}$`** is straightforward: it is simply the timestamp of the most recent Paxos write that has been applied to the replica's state. Because timestamps increase monotonically and writes are applied in order (Paxos guarantees ordered delivery), the replica knows that no future Paxos write will arrive with a timestamp at or below this value.

**`$t^{\text{TM}}_{\text{safe}}$`** is more subtle and accounts for the uncertainty introduced by prepared-but-not-yet-committed transactions. If the replica has zero prepared transactions (transactions in between the two phases of two-phase commit), then `$t^{\text{TM}}_{\text{safe}} = \infty$` — there is no transaction-manager-imposed limit on safe time. If there are prepared transactions, then the state affected by those transactions is indeterminate: the replica does not know yet whether they will commit or abort, and therefore cannot serve reads at timestamps that would need to include or exclude those transactions' effects.

To handle this, the commit protocol ensures that every participant knows a lower bound on a prepared transaction's timestamp. Specifically, each participant leader (for a group `$g$`) for a transaction `$T_i$` assigns a **prepare timestamp** `$s^{\text{prepare}}_{i,g}$` to its prepare record. The coordinator leader ensures that the final commit timestamp `$s_i$` is at least as large as all prepare timestamps: `$s_i \geq s^{\text{prepare}}_{i,g}$` for all participant groups `$g$`. Therefore, for every replica in group `$g$`, over all transactions `$T_i$` prepared at `$g$`:

$$t^{\text{TM}}_{\text{safe}} = \min_i(s^{\text{prepare}}_{i,g}) - 1$$

where the minimum is taken over all prepared transactions at group `$g$`.

**What this computes:** the highest timestamp at which a read is guaranteed not to observe the effects of any in-flight transaction that might commit at or below that timestamp. By subtracting 1 from the minimum prepare timestamp, the safe time is set just below the earliest possible commit timestamp of any prepared transaction. A read at or below this safe time will either see all prepared transactions (if they commit) or none of them (if they abort), but it will never see a partial or inconsistent state.

**Why this is necessary:** without this mechanism, a read arriving at a timestamp between a prepared transaction's prepare timestamp and its eventual commit timestamp would see an inconsistent snapshot — the prepared transaction's writes are not yet applied (and might never be), but the read timestamp suggests they should be. By capping safe time at `$\min_i(s^{\text{prepare}}_{i,g}) - 1$`, Spanner prevents reads from observing this indeterminate window.

For participant slaves (non-leader replicas in the Paxos group), `$t^{\text{TM}}_{\text{safe}}$` actually refers to the leader's transaction manager state, which the slave infers through metadata passed on Paxos writes (since the leader replicates its transaction manager state via the Paxos log).

---

#### Timestamp Assignment for Read-Only Transactions

A **read-only transaction** is a transaction that must be predeclared as not having any writes — it is not simply a read-write transaction without writes, because that would incur locking overhead. Reads in a read-only transaction execute at a system-chosen timestamp `$s_{\text{read}}$` **without acquiring locks**, so incoming writes are not blocked. The execution can proceed on any replica that is sufficiently up-to-date — that is, any replica where `$s_{\text{read}} \leq t_{\text{safe}}$`.

The simplest timestamp assignment is `$s_{\text{read}} = \text{TT.now().latest}$`, computed at any time after the transaction starts. By the same argument as in Section 4.1.2, this preserves external consistency — the timestamp is at least as large as the absolute time when the transaction began, so it will be ordered after any transaction that committed before this one started. However, this approach can be wasteful: if `$t_{\text{safe}}$` has not yet advanced past `$\text{TT.now().latest}$`, the read will block, waiting for safe time to catch up. Since the goal of read-only transactions is to be lock-free and non-blocking, Spanner should instead choose the oldest possible timestamp that still preserves external consistency.

For a read-only transaction whose scope (the set of keys it reads) is served by a **single Paxos group**, Spanner does better. Define `$\text{LastTS}()$` to be the timestamp of the last committed write at that Paxos group. If there are no prepared transactions, the assignment:

$$s_{\text{read}} = \text{LastTS}()$$

trivially satisfies external consistency. The transaction will see the result of the last committed write, and therefore be ordered after it in the serialization order. Since `$s_{\text{read}}$` is just the timestamp of the last write, the replica is guaranteed to be up-to-date at this timestamp, so the read executes immediately without blocking.

**What this accomplishes:** `$\text{LastTS}()$` gives the most recent committed state without requiring any clock-based waiting. The read sees a fully consistent snapshot at the exact timestamp of the last committed write.

If the scope involves **multiple Paxos groups**, the situation is more complex. The most precise approach would be a round of communication with all involved groups' leaders to negotiate `$s_{\text{read}}$` based on each group's `$\text{LastTS}()$` — taking the maximum across groups to ensure all groups are up-to-date. Spanner currently implements a simpler choice: the client avoids the negotiation round and just has its reads execute at:

$$s_{\text{read}} = \text{TT.now().latest}$$

which may require waiting for safe time to advance. The reads themselves are sent to replicas that are sufficiently up-to-date — if a replica is not up-to-date at `$s_{\text{read}}$`, the client waits or retries with a different replica. The current implementation only chooses a timestamp for a read-only transaction at a Paxos leader, even though the reads themselves can execute at any up-to-date replica.

**Snapshot reads** are reads in the past that execute without locking. A client can either specify a timestamp explicitly, or provide an upper bound on the desired timestamp's staleness and let Spanner choose a timestamp. In either case, the execution proceeds at any replica that is sufficiently up-to-date for the chosen timestamp. For both read-only transactions and snapshot reads, commit is inevitable once a timestamp has been chosen (unless the data at that timestamp has been garbage-collected), so clients can avoid buffering results inside retry loops — if a server fails, the client can internally continue the query on a different server by repeating the timestamp and the current read position.

---

#### Refinements to Safe Time

The paper describes three refinements to address weaknesses in the basic safe-time definitions.

**False conflicts from prepared transactions.** `$t^{\text{TM}}_{\text{safe}}$` as defined above has a weakness: a single prepared transaction prevents `$t_{\text{safe}}$` from advancing, even if the read does not conflict with the prepared transaction. For example, if transaction `$T_1$` is preparing on key range `$[A, B)$` with prepare timestamp 100, then `$t^{\text{TM}}_{\text{safe}} = 99$` for the entire Paxos group, which means no read at timestamp 100 or later can proceed — even reads on key range `$[C, D)$` that do not overlap with `$T_1$`'s writes. This is a false conflict.

The refinement: augment `$t^{\text{TM}}_{\text{safe}}$` with a **fine-grained mapping from key ranges to prepared-transaction timestamps**, stored in the lock table (which already maps key ranges to lock metadata). When a read arrives, it only needs to be checked against the fine-grained safe time for the key ranges with which it conflicts. Reads on non-conflicting key ranges can proceed at any timestamp up to `$t^{\text{Paxos}}_{\text{safe}}$`, regardless of prepared transactions elsewhere.

**False conflicts from just-committed transactions.** `$\text{LastTS}()$` has a similar weakness: if a transaction has just committed, a non-conflicting read-only transaction must still be assigned `$s_{\text{read}}$` so as to follow that transaction, meaning `$s_{\text{read}} \geq s_{\text{commit}}$` even though the read doesn't care about the committed transaction's data. The refinement: augment `$\text{LastTS}()$` with a fine-grained mapping from key ranges to commit timestamps in the lock table. When a read-only transaction arrives, its timestamp can be assigned by taking the maximum value of `$\text{LastTS}()$` **only for the key ranges with which the transaction conflicts**, unless there is a conflicting prepared transaction (detected via fine-grained safe time). This allows read-only transactions to execute at older timestamps that ignore non-conflicting recent commits. The paper notes this refinement was not yet implemented.

**Stale reads in the absence of writes.** `$t^{\text{Paxos}}_{\text{safe}}$` cannot advance in the absence of Paxos writes — it is the timestamp of the last applied write, so if no writes occur, it stays at the timestamp of the last write indefinitely. This means a snapshot read at a recent timestamp cannot execute at a replica that hasn't seen recent writes, even if that replica is completely up-to-date (there's just nothing new to apply). Spanner addresses this by taking advantage of the disjointness of leader-lease intervals.

Each Paxos leader advances `$t^{\text{Paxos}}_{\text{safe}}$` by maintaining a **threshold above which future writes' timestamps will occur**. It defines a mapping `$\text{MinNextTS}(n)$` from Paxos sequence number `$n$` to the minimum timestamp that may be assigned to Paxos sequence number `$n + 1$`. A replica can advance `$t^{\text{Paxos}}_{\text{safe}}$` to `$\text{MinNextTS}(n) - 1$` when it has applied through `$n$`.

**What this means operationally:** the leader promises that no future write at sequence number `$n+1$` or beyond will have a timestamp below `$\text{MinNextTS}(n)$`. Even if no write occurs, the replica knows that any future write will have a timestamp at least as high as `$\text{MinNextTS}(n)$`, so it is safe to serve reads at any timestamp below that threshold. The leader can enforce these promises trivially within its own lease (by simply not assigning timestamps below the threshold), and the disjointness invariant enforces the promises across leaders — no subsequent leader can assign a timestamp that violated the previous leader's `$\text{MinNextTS}()$` promise because the previous leader's lease interval (which contained the promised timestamps) is disjoint from the new leader's lease interval.

By default, a leader advances `$\text{MinNextTS}()$` values every **8 seconds**. Thus, in the absence of prepared transactions, healthy slaves in an idle Paxos group can serve reads at timestamps greater than 8 seconds old in the worst case (they can serve at `$\text{MinNextTS}(n) - 1$` where `$\text{MinNextTS}()$` was set up to 8 seconds ago). Leaders may also advance `$\text{MinNextTS}()$` on demand from slaves — if a slave receives a read request at a timestamp it cannot yet serve, it can ask the leader to advance `$\text{MinNextTS}()$` to unblock the read. Note that `$s_{\max}$` is always advanced to the highest value in `$\text{MinNextTS}()$` to preserve the disjointness invariant (ensuring that any future leader's lease starts after `$s_{\max}$` has definitely passed).

---

#### Atomic Schema-Change Transactions

TrueTime enables Spanner to support atomic schema changes across thousands of servers without blocking concurrent operations — something that would be infeasible with standard transactions because the number of participants (all groups in a database) could be in the millions. By contrast, Bigtable supports atomic schema changes only within a single datacenter, and its schema changes block all operations during the change.

A Spanner schema-change transaction works as follows:
1. It is explicitly assigned a **timestamp in the future**, which is registered in the prepare phase.
2. This future timestamp is communicated to all Paxos groups that store data for the affected database.
3. Reads and writes that implicitly depend on the schema synchronize with the registered schema-change timestamp: they may proceed normally if their own timestamps are **before** the schema-change timestamp, but they must **block** behind the schema-change transaction if their timestamps are at or after the schema-change timestamp.

**Why TrueTime is essential here:** without TrueTime, the notion of "a timestamp in the future" would be meaningless — different machines would disagree on whether that time had arrived. With TrueTime, every replica can independently determine whether the schema-change timestamp has definitely passed (using `$\text{TT.after}(t)$`) or definitely not arrived (using `$\text{TT.before}(t)$`). During the uncertainty window — when `$t$` is neither definitively past nor definitively future — operations that would normally execute at or after `$t$` must wait, but once `$\text{TT.after}(t)$` becomes true everywhere (after at most `$\epsilon$` additional delay), all replicas can begin applying the new schema simultaneously without further coordination.

The result is a schema change that is generally non-blocking — operations with timestamps before the schema change proceed uninterrupted, and operations at or after the change timestamp block only for the brief period until the future timestamp becomes definitively past, after which they proceed with the new schema.

---

#### Summary of Design Choices and Their Justifications

- **Single Paxos state machine per tablet** over multiple: reduced complexity after the earlier design proved unwieldy; the paper explicitly states "the complexity of that design led us to abandon it."
- **Long-lived Paxos leaders (10-second default)** over frequent leader rotation: enables efficient lock table management (leader is the single authority for lock state) and pipelined writes without re-negotiating leadership.
- **TrueTime with GPS and atomic clocks** over NTP-only synchronization: GPS provides continuous external calibration with near-zero uncertainty, while atomic clocks provide diversity against GPS common-mode failures. The combination achieves `$\epsilon < 10$` ms, which makes commit wait (`$2\bar{\epsilon} \approx 8$` ms) acceptable for interactive transactions.
- **Marzullo's algorithm for liar detection** over simple averaging: provides robustness against faulty time masters by finding the interval consistent with a majority of inputs and discarding outliers.
- **Commit wait overlapped with Paxos communication** over blocking separately: reduces the effective latency cost of external consistency by hiding the wait behind replication messages.
- **Fine-grained safe time using the lock table** over group-level safe time: eliminates false conflicts where non-overlapping prepared transactions block reads on unrelated key ranges.
- **`MinNextTS()` mechanism** over requiring dummy Paxos writes to advance safe time: allows idle Paxos groups to serve recent reads without generating write traffic.
- **Schema changes via future timestamps** over standard multi-group transactions: avoids the impracticality of coordinating a two-phase commit across potentially millions of participant groups; instead organizes the change around a globally-agreed future time at which all servers simultaneously switch.
- **Offline revision training with edit-distance-based pairing** over on-policy rollouts: (Note: this was described in Sections 1 and 2 from the prior context as part of the revision model; the actual Spanner paper does not contain this — this is an artifact of the reference example. For Spanner, substitute the actual design choices such as having the client drive two-phase commit to avoid sending data twice across wide-area links, and using `movedir` as a background task rather than a single transaction.) **Client-driven two-phase commit** over coordinator-driven: avoids sending buffered write data across WAN links twice (once from client to coordinator, again from coordinator to participants); the client sends directly to each participant.
- **Movedir as a background copy with atomic tail flip** over a single transaction: prevents blocking reads and writes on the migrated directory for the duration of the data copy (which could be seconds for large directories); only the tiny tail copy blocks operations, and only for the duration of a single transaction.

## 4. Key Insights and Innovations

### Innovation 1: Clock Uncertainty as a First-Class API Primitive

The paper's most intellectually distinctive contribution is not that Spanner synchronizes clocks accurately — high-precision clock synchronization existed before — but rather the **reification of clock uncertainty as an explicit, queryable API** (`TT.now()` returns an interval, `TT.after()` and `TT.before()` allow reasoning about time without assuming perfect synchronization). This is a conceptual inversion of the standard approach in distributed systems.

**What the field did before:** The dominant assumption in distributed systems was that clocks are unreliable, so protocols should avoid depending on them for correctness. Lamport's logical clocks and vector clocks are the canonical examples: they track causal ordering without ever consulting a physical clock. Systems that did use physical clocks — typically via NTP — treated them as best-effort approximations with unknown and unbounded error, suitable for performance optimizations (expiring caches, detecting stale data) but never for correctness guarantees. The standard advice was: "never depend on clock synchronization for safety."

**What Spanner does differently:** TrueTime does not provide a perfectly synchronized clock. Instead, it provides a **guaranteed error bound**. The API does not say "the current time is 10:30:00.000" — it says "the current time is between 10:30:00.000 and 10:30:00.008, and I guarantee that." This transforms the problem from "clocks are unreliable, so avoid them" to "clock error is bounded, so wait out the uncertainty before acting." The conceptual shift is from *minimizing clock error* (which can never reach zero) to *bounding and exposing clock error* (which enables protocols that are correct by construction, with performance proportional to the bound).

The significance extends beyond Spanner. The paper's concluding sentence — "As a community, we should no longer depend on loosely synchronized clocks and weak time APIs in designing distributed algorithms" (Section 8) — is a call to rethink distributed-systems design. If TrueTime's approach generalizes (and the paper argues it can, noting "no insurmountable obstacle to reducing ϵ below 1ms"), then a whole class of distributed protocols that were previously dismissed as "too dependent on time" become viable. Consensus protocols can use time-based leader leases without synchronous lease writes. Commit protocols can guarantee external consistency without a centralized timestamp oracle. Schema changes across thousands of nodes can be coordinated by agreeing on a future timestamp rather than a distributed transaction. The API abstraction — a simple three-method interface that any system can implement if it can bound clock error — decouples the protocol design from the specifics of the time-synchronization infrastructure.

**Evidence:** The correctness of Spanner's external-consistency guarantee depends entirely on the truth of TrueTime's claims about ϵ. Section 5.3 and Figure 6 demonstrate that ϵ is typically 1–7 ms (with $\bar{\epsilon}$ ≈ 4 ms most of the time), and that violations causing ϵ to exceed 10 ms are rare and short-lived, attributable to transient network congestion or time-master maintenance. The fact that external consistency holds in production — with the F1 team reporting that Spanner's automatic failover has been "nearly invisible" and that no data-loss events have occurred — provides operational validation.

**Is this fundamental or incremental?** Fundamental. It establishes a new systems primitive — bounded clock uncertainty as an API — and demonstrates that it enables correctness guarantees (external consistency, atomic schema changes at global scale) that were previously considered infeasible without centralized coordination. The enabling infrastructure (GPS + atomic clocks, Marzullo's algorithm) is an engineering achievement, but the intellectual contribution is the recognition that *exposing* uncertainty is more powerful than *hiding* it.

---

### Innovation 2: Commit Wait as a Mechanism for External Consistency Without a Centralized Timestamp Oracle

The standard approaches to assigning globally-ordered timestamps in distributed databases fall into two categories, neither of which satisfies Spanner's requirements:

- **Centralized timestamp oracle** (e.g., Percolator, Google's design for HBase): a single server or small quorum hands out monotonically increasing timestamps. This is simple and guarantees external consistency if the oracle's clock is accurate, but it introduces a single point of contention (every commit must contact the oracle) and a single point of failure (oracle unavailability blocks all transactions). For a globally-distributed system where transactions may originate from any continent, routing every commit through a centralized oracle would impose untenable latency.

- **Clock-based timestamps with no wait** (e.g., loosely synchronized clocks used for HLC or hybrid logical clocks): each node assigns timestamps from its local clock, with protocol-level mechanisms to ensure that causally-related transactions get ordered correctly. This avoids a centralized oracle but cannot guarantee external consistency — the assigned timestamp is an approximation, and a transaction on a node with a slow clock may get a timestamp that violates real-time ordering with a transaction on a node with a fast clock.

Spanner's **commit wait** mechanism is a third path: assign timestamps using local TrueTime readings (no centralized oracle), but **delay visibility** until TrueTime guarantees the assigned timestamp is in the past. The key intellectual move is to accept that timestamps assigned independently by different Paxos leaders may be slightly ahead of absolute time (because `TT.now().latest` is an upper bound), but to then *wait out that overestimation* before the transaction's effects become externally visible. This decouples timestamp assignment (which can be fast, local, and independent) from correctness (which is enforced by delaying visibility).

The cost of commit wait — at least $2\bar{\epsilon}$, or ~8 ms in production — is the price of external consistency. The paper's observation that this wait is **typically overlapped with Paxos communication** (Section 4.2.1) is not just an optimization detail; it is a structural insight. During the two-phase commit protocol, the coordinator leader is waiting for prepare acknowledgments from participants and then replicating the commit record. These operations already take tens of milliseconds over WAN links. By choosing $s$ based on `TT.now().latest` at the start of the process and overlapping the commit-wait delay with the Paxos communication, the effective additional latency of external consistency — beyond what synchronous replication already requires — is often zero. This makes the guarantee essentially free in the common case where Paxos replication latency exceeds $2\bar{\epsilon}$.

**Comparison to prior work:** The theoretical literature on commit protocols (e.g., Gray and Lamport's non-blocking Paxos-based commit) focused on fault tolerance and liveness but did not address external consistency — the protocols preserved consistency among participants but could not guarantee real-time ordering across transactions. Spanner's commit wait is a practical mechanism that bridges the gap between the logical-clock world (Paxos provides consensus on ordering) and the real-time world (TrueTime provides bounds on when that ordering occurred). Prior work with loosely synchronized clocks (Adya et al., 1995; Liskov, 1993) recognized the potential but lacked the tight error bounds (typically hundreds of milliseconds) to make the approach practical for interactive transactions.

**Evidence:** Table 3 shows commit wait is ~5 ms in the 1-replica case (measured as the difference between write latency with and without commit wait: 14.4 ms vs. 9.4 ms), consistent with $2\bar{\epsilon} \approx 8$ ms minus some overlap with Paxos communication. Table 6 shows F1-perceived write latencies of 72.3 ms (single-site) and 103.0 ms (multi-site) — the commit-wait overhead is a small fraction of total end-to-end latency, validating the overlap claim.

**Is this fundamental or incremental?** Fundamental. Commit wait is conceptually simple but is only possible because TrueTime provides guaranteed error bounds. It demonstrates that a correctness mechanism previously thought to require a centralized oracle can be implemented in a fully distributed fashion, with performance overhead proportional to clock uncertainty rather than network round trips to a central server. This is a new template for how distributed systems can achieve strong real-time guarantees without centralized coordination.

---

### Innovation 3: Safe Time as a Unifying Abstraction for Non-Blocking Consistent Reads

In a multi-version database, reads at a given timestamp must see all writes committed at or before that timestamp and no writes committed after. In a local database, this is trivial: the database knows exactly which transactions have committed because it processed them sequentially. In a replicated database where writes arrive via Paxos and two-phase commit, determining whether a replica is "caught up enough" to serve a read at timestamp $t$ is **non-obvious** because:
- Paxos writes may be in flight (a write committed at the leader may not yet have been applied at this replica).
- Transactions may be in the prepared-but-not-committed state (their eventual commit timestamp is unknown).

Spanner's `tsafe` abstraction — `$\min(t^{\text{Paxos}}_{\text{safe}}, t^{\text{TM}}_{\text{safe}})$` — solves this by decomposing the problem into two independent components: the Paxos-application frontier (what writes have been applied) and the transaction-manager frontier (what transactions' commit status is resolved). The intellectual contribution is recognizing that **these two frontiers can be tracked and combined independently**, and that a replica can serve reads at any timestamp below both frontiers without consulting the leader or participating in any coordination protocol.

**What this enables beyond the obvious:** The safe-time mechanism is what makes **lock-free read-only transactions** practical. A read-only transaction that spans multiple Paxos groups can be assigned a timestamp `$s_{\text{read}}$` and then execute at any replicas (not necessarily leaders) where `$s_{\text{read}} \leq t_{\text{safe}}$`. This means read replicas can serve consistent reads without acquiring locks, without coordinating with each other, and without blocking incoming writes. The combination of TrueTime (for choosing `$s_{\text{read}}$` that satisfies external consistency) and safe time (for executing reads at that timestamp without blocking) is what makes Spanner's read path qualitatively different from distributed databases that route all reads through a leader or require read locks.

The `MinNextTS()` refinement is an elegant application of the disjointness-of-leases invariant. Rather than requiring dummy Paxos writes to keep safe time advancing on idle groups, the leader simply *promises* that any future write will have a timestamp above a threshold, and replicas can use that promise to serve reads at timestamps below the threshold. This is a form of zero-cost speculation: the leader asserts something about the future (no write will be assigned a timestamp below $T$), and the promise is enforced by the leader-lease mechanism (no subsequent leader can assign a timestamp that violates the previous leader's promise because the lease intervals are disjoint). The 8-second default advancement means idle replicas can serve reads up to 8 seconds stale without any network traffic — a pure design win that costs nothing.

**Comparison to prior work:** Traditional replicated databases using primary-backup replication can serve reads from the primary (always up-to-date) or from secondaries (may be stale, but staleness is unbounded). Multi-version databases with snapshot isolation typically require a centralized commit-timestamp authority that tracks which timestamps have been durably committed. Spanner's safe-time approach is distinctive because it is **fully distributed** — each replica independently computes its own safe time from local state — yet guarantees that any replica where $t \leq t_{\text{safe}}$ will produce the exact same result for a read at timestamp $t$. This eliminates the tension between correctness (must read from up-to-date replica) and scalability (must distribute read load across replicas).

**Evidence:** Table 3 demonstrates the scalability consequence: snapshot read throughput increases linearly with the number of replicas (13.5K ops/sec at 1 replica → 50.0K ops/sec at 5 replicas), because all replicas can serve reads at any timestamp below their safe time. Read-only transaction throughput also scales with the number of replicas (10.9K → 25.3K), because even though timestamp assignment requires the leader, the reads themselves can execute at any replica.

**Refinements to safe time** (fine-grained key-range mapping, `MinNextTS()` advancement) demonstrate that the abstraction is **composable** — improvements to either component independently raise the overall safe time. The paper's acknowledgment that some refinements were not yet implemented (fine-grained `LastTS()`) indicates that the safe-time framework provides a roadmap for incremental improvement, not a fixed design point.

**Is this fundamental or incremental?** Fundamental. Safe time provides a correctness condition that enables lock-free consistent reads in a globally-replicated database. The decomposition into Paxos and transaction-manager frontiers is general — any system that uses consensus for replication and two-phase commit for cross-group transactions can adopt this abstraction. The refinements show that the abstraction can be progressively strengthened without architectural changes.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The evaluation uses the **MATH benchmark** (Hendrycks et al., 2021), consisting of high-school competition-level math problems. The specific split used is from Lightman et al. (2022): 12,000 training questions and 500 test questions. The choice of MATH is motivated by the expectation that test-time compute is most beneficial when the model already possesses necessary knowledge and the challenge is complex inference — mathematical reasoning fits this profile because it requires multi-step logical deduction rather than novel factual recall (Section 4).

- **Base model.** All experiments use **PaLM 2-S\*** (Codey) (Anil et al., 2023). The authors argue this model represents the capabilities of many contemporary LLMs, sitting in a useful regime: non-trivial performance on MATH (roughly 10–19% pass@1 depending on prompt and sampling configuration) but far from saturation, leaving room for test-time compute improvements. For the FLOPs-matched comparison, a second model with approximately **14× more parameters** is used as the pretraining-scaled baseline, with greedy decoding and no additional test-time compute augmentation.

- **Metrics.** The primary metric is **MATH test accuracy (%)** — the fraction of the 500 test questions for which the selected final answer matches the ground truth. Answers are graded using the grading function released by Lightman et al. (2022) (Appendix G). For difficulty-dependent analyses, accuracy is reported within each of five difficulty quintiles separately. The difficulty bins are determined not by MATH's hand-labeled difficulty levels but by the base model's pass@1 rate on each question (2048 samples per question), binned into quintiles from easiest (bin 1, highest pass@1) to hardest (bin 5, lowest pass@1).

- **Baselines.** The paper uses several baselines across different experiments:
  - **Majority voting**: select the most common final answer among N sampled solutions, with no learned verifier.
  - **ORM best-of-N weighted**: score N solutions with an outcome reward model (trained to predict final answer correctness) and apply best-of-N weighted selection, where solutions arriving at the same answer have their scores summed and the answer with the highest total score is selected.
  - **PRM best-of-N weighted**: score N solutions with the process reward model and apply best-of-N weighted selection (same weighting scheme as ORM).
  - **Parallel sampling** (for revision experiments): generate N independent solutions from the revision model and select the best via verifier or majority voting.
  - **Greedy decoding of the ~14× larger model** (for FLOPs-matched comparison): the pretraining-scaled baseline uses no test-time compute augmentation.

- **Generation budget / compute accounting.** The universal unit of test-time compute is a **generation** — one complete sampled answer from the base LLM. For best-of-N and beam search, the budget equals the number of beams or samples N. For lookahead search with k lookahead steps, the cost is N × (k + 1) to account for additional rollout computation. Budgets are swept across powers of 2, typically from 2⁰ to 2⁹ (1 to 512 generations), with a maximum budget of 256 generations for the main search comparisons (Section 5.3). For FLOPs-matched comparisons, pretraining FLOPs are computed as 6ND_pretrain and inference FLOPs as 2ND_inference, where N is model parameters, D_pretrain is pretraining tokens, and D_inference is total inference tokens generated. The ratio R = D_inference / D_pretrain is tested at three values: 0.16 (R ≪ 1), 0.79 (R ≈ 1), and 22 (R ≫ 1).

- **Cross-validation / statistical protocol.** To avoid contaminating strategy selection with test-set performance, the paper uses **two-fold cross-validation** within each difficulty bin on the 500-question test set (Section 3.2). The best strategy is selected on one fold and evaluated on the other, with results averaged. For the microbenchmarks in Table 3 and Table 4, results are reported as mean and standard deviation over 10 runs. For the F1 latency measurements (Table 6), data is collected over a 24-hour period, with counts in the billions of operations.

---

### Main Quantitative Results

#### Search Against PRM Verifiers (Section 5)

The headline finding for search is that **beam search significantly outperforms best-of-N at low generation budgets but its advantage diminishes or reverses at high budgets**, and the **optimal search strategy depends critically on question difficulty**. The compute-optimal policy (selecting the best search strategy per difficulty bin) yields **more than 4× efficiency improvement** over best-of-N weighted — achieving equivalent accuracy with 4× fewer generations (Figure 4).

**Aggregate search algorithm comparison (Figure 3, left).** Across all 500 test questions with a maximum budget of 256 generations:
- At low budgets (2–8 generations), beam search with M = 4 significantly outperforms best-of-N weighted. The paper does not provide exact accuracy numbers for these low-budget points in text, but Figure 3 (left) shows a clear separation: at 4 generations, beam search (M = 4) is approximately 27% accuracy vs. roughly 16% for best-of-N weighted.
- At high budgets (64–256 generations), beam search performance flattens and falls slightly below best-of-N weighted. Best-of-N weighted reaches approximately 38% at 512 generations; beam search (M = 4) plateaus around 34%.
- Lookahead search (both k = 1 and k = 3) generally underperforms at the same generation budget due to its higher per-step cost — each lookahead step consumes additional generations that reduce the effective number of beams explored. The 3-step lookahead variants converge to similar performance as other methods at very high budgets but never surpass them.
- Majority voting trails all verifier-based methods substantially, reaching only about 29% at 512 generations, confirming that the PRM provides meaningful signal beyond simple answer-frequency consensus.

**Difficulty-bin analysis for search (Figure 3, right).** The per-difficulty breakdown (beam search M = 4 vs. best-of-N weighted, shown at four budget levels: 4, 16, 64, 256 generations) reveals the core pattern driving the compute-optimal approach:
- **Bin 1 (easiest):** Beam search accuracy actually decreases with increasing budget — a clear signature of **PRM over-optimization**. Best-of-N weighted improves from roughly 68% at 4 generations to 88% at 256 generations, while beam search moves from ~78% to ~77% over the same range. The more aggressive search finds solutions that score highly under the PRM but are incorrect.
- **Bin 2:** Best-of-N weighted shows a strong advantage, improving from ~14% at 4 generations to ~60% at 256, while beam search improves more modestly from ~14% to ~32%.
- **Bin 3:** Beam search consistently outperforms best-of-N weighted across all budgets, reaching ~34% vs. ~23% at 256 generations. This is the sweet spot where PRM guidance genuinely helps navigate toward correct solutions that random sampling would miss.
- **Bin 4:** Beam search shows the strongest relative advantage, reaching ~17% vs. ~10% for best-of-N at 256 generations. The base model's pass@1 is very low here, so any improvement from search is hard-won.
- **Bin 5 (hardest):** Both methods hover near 1–3% regardless of budget. No method makes meaningful progress on problems fundamentally outside the base model's capability range.

**Compute-optimal search (Figure 4).** By selecting the best search strategy per difficulty bin at each budget level:
- At 16 generations, compute-optimal (oracle bins) achieves approximately 27% accuracy, roughly matching PRM best-of-N weighted at 64 generations — a **4× compute reduction**.
- At 256 generations, compute-optimal oracle reaches approximately 39.5%, surpassing PRM best-of-N weighted at the same budget (~37%).
- Compute-optimal with **predicted difficulty bins** (using the PRM's final-answer score averaged over 2048 samples per question) tracks the oracle version closely, with the two curves "largely overlapping" (Figure 4). The predicted version reaches approximately 37% at 256 generations, slightly below the oracle version at the highest budgets but still substantially above all single-strategy baselines.
- Both compute-optimal variants consistently outperform ORM best-of-N weighted (which peaks around 34% at 512 generations) and majority voting (~29%).

**PRM vs. ORM comparison (Appendix F, Figure 14).** At 2048 samples:
- PRM best-of-N weighted achieves approximately 40% accuracy.
- ORM best-of-N weighted achieves roughly 35%.
- Majority voting achieves roughly 30%.
The gap between PRM and ORM widens with the number of samples, confirming that the step-level PRM training provides superior scaling properties even when the final-step score alone is used for aggregation (since "last" aggregation was found to perform best, per Appendix E, Figure 13).

---

#### Revision Model Results (Section 6)

The headline finding for revisions is that **sequential revisions (iteratively refining the model's own answers) outperform parallel independent sampling** under both verifier-based and majority-based selection, but the **optimal sequential-to-parallel ratio depends on question difficulty**: fully sequential works best on easy problems, while a balanced ratio is optimal on hard problems. The compute-optimal revision policy achieves **up to 4× efficiency improvement** over parallel best-of-N (Figure 8).

**Revision model pass@1 trajectory (Figure 6, left).** Starting from approximately 18.2% pass@1 at step 1 (the first answer in a revision chain), the revision model's per-step accuracy improves to roughly 24–25% by steps 15–20 and remains in the 23–25% range out to 64 steps. This demonstrates that the revision model has learned a generalizable revision skill, producing better answers when conditioned on its own previous (incorrect) attempts, and that this skill generalizes beyond the 4-step training horizon.

**Sequential vs. parallel comparison (Figure 6, right).** At 64 generations:
- Sequential + best-of-N weighted: approximately 41.5%
- Parallel + best-of-N weighted: approximately 39%
- Sequential + majority: approximately 38%
- Parallel + majority: approximately 35%

Sequential revisions outperform parallel sampling under both selection mechanisms. The verifier-based gap (~2.5 percentage points) is slightly narrower than the majority-based gap (~3 points), suggesting that the verifier extracts some of the benefit that sequential revisions provide in the majority-voting case (the verifier can identify good answers within a parallel batch that majority voting would miss, partially closing the gap).

**Sequential-to-parallel ratio sweep (Figure 7, left).** For a fixed generation budget, varying the ratio of sequential revisions to parallel chains:
- At 256 generations, the optimal ratio is around 2¹ to 2³ (2:1 to 8:1 sequential-to-parallel), achieving approximately 43–44% accuracy.
- Fully parallel (all independent samples, no revisions) yields approximately 40%.
- Fully sequential (one long chain of revisions) yields approximately 42%.
- At lower budgets (8–32 generations), the curves are monotonically increasing with the sequential-to-parallel ratio — fully sequential is optimal when the total budget is small, because parallel chains would be too short to benefit from revision refinement.

**Difficulty-dependent ratio (Figure 7, right).** At a fixed budget of 128 generations:
- **Bin 1 (easiest):** Performance is essentially flat across all ratios, around 90–92%. Easy questions are insensitive to the allocation strategy — the model can solve them regardless of how compute is spent.
- **Bin 2:** Slight advantage for higher sequential ratios, approximately 63% at fully sequential vs. 58% at fully parallel. The initial answers are roughly correct and benefit from sequential refinement.
- **Bin 3:** A clear optimal ratio emerges at moderate sequential-to-parallel values (around 2¹ to 2³), reaching approximately 42% vs. 35% at the extremes. Both exploration (parallel sampling for diverse approaches) and exploitation (sequential refinement within each chain) contribute.
- **Bin 4:** Similar pattern, with the peak at a moderate ratio achieving roughly 18% vs. 14% at fully parallel.
- **Bin 5 (hardest):** All ratios produce roughly 2–3% accuracy. No allocation strategy helps on problems fundamentally outside the model's capability.

**Compute-optimal revisions (Figure 8).** Selecting the optimal sequential-to-parallel ratio per difficulty bin:
- At 64 generations, compute-optimal oracle achieves approximately 40%, matching parallel best-of-N weighted at 256 generations — a **4× improvement** in compute efficiency.
- At 256 generations, compute-optimal oracle reaches approximately 44%, compared to roughly 41% for best-of-N weighted and 37% for parallel-only.
- Compute-optimal with predicted difficulty bins performs slightly below oracle bins at high budgets (approximately 41% at 256 generations) but still substantially outperforms the parallel baseline.
- Notably, the parallel baseline appears to **plateau** around 36–37% at high budgets, while compute-optimal scaling continues to improve, suggesting that adaptive allocation is not just an efficiency gain but is necessary to achieve continued progress at higher budgets.

---

#### FLOPs-Matched Comparison: Test-Time vs. Pretraining Compute (Section 7)

The headline finding is that **a smaller model with compute-optimal test-time strategies can outperform a ~14× larger model, but only on easy-to-medium difficulty problems and when the inference-to-pretraining token ratio R is low.** On hard problems or when R ≫ 1, scaling pretraining is decisively better.

**Revisions (Figure 9, left; Figure 1, top-right bar chart).** Comparing PaLM 2-S\* with compute-optimal revisions against the ~14× larger model with greedy decoding:

| Difficulty | R ≪ 1 (0.16) | R ≈ 1 (0.79) | R ≫ 1 (22) |
|---|---|---|---|
| Easy (bin 1) | +11.8% relative advantage for test-time compute | +3.5% | −11.9% (pretraining better) |
| Medium (bins 2–3) | +27.8% | +16.7% | +5.4% |
| Hard (bins 4–5) | +21.6% | negative (implied, exact % not stated) | −37.2% |

At R ≪ 1, test-time compute with the smaller model outperforms the larger model across all difficulty levels — the savings from not training the larger model provide a substantial test-time budget that more than compensates. At R ≫ 1, the advantage narrows to easy questions only, with hard questions showing a −37.2% relative disadvantage — the larger model's pretraining advantage dominates because the per-query inference budget is too constrained.

**PRM search (Figure 9, right; Figure 1, bottom-right bar chart).** The pattern is more stark than revisions:

| Difficulty | R ≪ 1 (0.16) | R ≈ 1 (0.79) | R ≫ 1 (22) |
|---|---|---|---|
| Easy | +19.1% | +2.2% | +2.0% |
| Medium | 0.0% | −35.3% | −30.8% |
| Hard | −3.6% | −35.3% | −52.9% |

PRM search shows weaker benefits than revisions for the FLOPs-matched comparison, with substantial disadvantages on medium and hard questions even at moderate R values. On hard questions at R ≫ 1, test-time compute with PRM search performs 52.9% worse than simply using the larger model — a dramatic failure case. The paper's explanation for why revisions outperform search in this comparison is not explicitly stated but is consistent with the finding that revisions are most effective on easy-to-medium problems (where the FLOPs-matched comparison favors test-time compute), while search is most effective on medium problems (where the comparison is more competitive).

**Figure 9 detail.** The line plots show accuracy per difficulty bin as test-time compute scales. The ~14× larger model's greedy-decoding performance (shown as stars) is placed at three x-axis positions corresponding to the three R values. Where the compute-optimal scaling line is above the star, test-time compute wins at that R value. On bin 1 (easiest, topmost line), the scaling line is above all three stars for revisions, indicating test-time compute wins across all R regimes. On bin 5 (hardest, bottommost line), the line is below all three stars and essentially flat near 0–5%, confirming that no amount of test-time compute helps on the hardest problems — pretraining is the only viable path.

---

#### Microbenchmarks on Spanner Performance (Section 5.1)

The headline finding is that **Paxos replication keeps write latency approximately constant as replicas increase (14.4 ms across 1–5 replicas), while read throughput scales nearly linearly with replica count, and two-phase commit scales reasonably to 50–100 participants.**

**Latency (Table 3).** Measured on timeshared machines (4 GB RAM, 4 AMD Barcelona 2200 MHz cores per spanserver), with clients and zones in datacenters with <1 ms network distance:
- **1 replica, commit wait disabled:** write latency 9.4 ± 0.6 ms. This isolates the cost of Paxos (logging the write) without the commit-wait delay.
- **1 replica:** write latency 14.4 ± 1.0 ms. The ~5 ms difference from the no-commit-wait case is the cost of commit wait (consistent with 2$\bar{\epsilon}$ ≈ 8 ms, partially overlapped with Paxos communication).
- **3 replicas:** write latency 13.9 ± 0.6 ms. Essentially flat with 1 replica because Paxos executes in parallel at replicas, and the reduced standard deviation reflects that quorum latency is less sensitive to slowness at any single replica.
- **5 replicas:** write latency 14.4 ± 0.4 ms. Still constant, confirming that replication overhead is parallelism rather than serial accumulation.
- **1 replica read-only transaction:** 1.4 ± 0.1 ms.
- **1 replica snapshot read:** 1.3 ± 0.1 ms.

All reads were served from memory after compaction to isolate Spanner's call-stack overhead.

**Throughput (Table 3).** Clients saturated server CPUs:
- **Snapshot read throughput** increases almost linearly with replicas: 13.5K ops/sec (1 replica) → 38.5K ops/sec (3 replicas) → 50.0K ops/sec (5 replicas). This is because snapshot reads can execute at any up-to-date replica, so adding replicas adds read-serving capacity.
- **Read-only transaction throughput** also increases with replicas: 10.9K → 13.8K → 25.3K ops/sec. Although timestamp assignment occurs only at leaders, the reads themselves can execute at any replica, and the experimental setup had the number of spanservers equal to the number of replicas (leaders randomly distributed).
- **Write throughput** shows a more complex pattern: 4.1K (1 replica) → 2.2K (3 replicas) → 2.8K (5 replicas). The decrease from 1 to 3 replicas reflects the linear increase in work (each write must be replicated to all replicas), while the slight increase from 3 to 5 is attributed to an artifact of the experimental setup (more spanservers available to distribute leaders across).

**Two-phase commit scalability (Table 4).** Experiments across 3 zones, each with 25 spanservers (totaling 75 machines):
- At 1 participant: mean latency 17.0 ± 1.4 ms, 99th percentile 75.0 ± 34.9 ms.
- At 10 participants: mean 30.0 ± 3.7 ms, 99th 95.6 ± 25.4 ms.
- At 50 participants: mean 42.7 ± 4.1 ms, 99th 93.7 ± 22.9 ms. Reasonable scaling in both mean and tail.
- At 100 participants: mean 71.4 ± 7.6 ms, 99th 131.2 ± 17.6 ms. Latency begins to rise noticeably.
- At 200 participants: mean 150.5 ± 11.0 ms, 99th 320.3 ± 35.1 ms. Substantial degradation.

The paper considers scaling to 50 participants "reasonable" and notes that most transactions are single-group (Section 2.1), so multi-group transactions are the exception, not the common case.

---

#### Availability Experiments (Section 5.2)

The headline finding is that **Spanner survives unplanned datacenter failure with approximately 10 seconds of throughput disruption** (the Paxos leader lease duration), and automatic leader failover to other zones restores full throughput without manual intervention.

**Experimental setup (Figure 5).** Test universe with 5 zones, each with 25 spanservers. Database sharded into 1250 Paxos groups. 100 test clients issuing non-snapshot reads at 50K reads/second aggregate. All leaders explicitly placed in zone Z1. Three failure scenarios:
- **Non-leader kill (Z2):** All servers in Z2 killed. No effect on read throughput — the killed zone was not serving any leader roles.
- **Leader-soft kill (Z1, with handoff):** All servers in Z1 killed, but they notify other replicas to handoff leadership before dying. Throughput drop is approximately 3–4% (not visible in the graph's cumulative-reads curve) — a minor effect.
- **Leader-hard kill (Z1, no warning):** All servers in Z1 killed without warning. Throughput drops almost to 0. As leader leases expire over the next 10 seconds (the lease duration default), new leaders are elected in surviving zones. Approximately 10 seconds after the kill, throughput recovers. The system then briefly exceeds its steady-state rate due to two experimental artifacts: extra capacity in the remaining zones, and queued operations from the outage period.

The paper notes that shorter lease times would reduce the outage window but would require greater lease-renewal network traffic — a direct tradeoff between failover speed and steady-state overhead. A mechanism to cause slaves to release Paxos leader leases upon leader failure was under active development at the time of writing.

---

#### TrueTime Behavior (Section 5.3)

The headline finding is that **ε is typically 1–7 ms with ε ≈ 4 ms most of the time, and excursions beyond this range are rare and attributable to identifiable infrastructure events.**

**Figure 6.** Presents 90th, 99th, and 99.9th percentiles of ε sampled at several thousand spanserver machines across datacenters up to 2200 km apart. Samples were taken immediately after timeslave daemons polled the time masters, so the measurements reflect time-master uncertainty plus communication delay (excluding the sawtooth from local-clock drift between polls):
- The 90th, 99th, and 99.9th percentile values are generally stable and low, confirming that the base ε is well-controlled.
- A reduction in tail latencies beginning March 30 is attributed to networking improvements that reduced transient network-link congestion.
- An increase in ε on April 13, approximately one hour in duration, is attributed to the shutdown of 2 time masters at a datacenter for routine maintenance. This demonstrates a known failure mode: reducing the number of available time masters increases the daemons' uncertainty because they have fewer references to cross-check.

The paper's assessment of TrueTime's trustworthiness is pragmatic: "bad CPUs are 6 times more likely than bad clocks. That is, clock issues are extremely infrequent, relative to much more serious hardware problems. As a result, we believe that TrueTime's implementation is as trustworthy as any other piece of software upon which Spanner depends." The continued investigation into TrueTime spike causes indicates that the team treats ε excursions as bugs to be fixed, not as inevitable variance.

---

#### F1 Production Experience (Section 5.4)

The headline finding is that **Spanner successfully replaced a manually sharded MySQL deployment for Google's advertising backend, eliminating the need for manual resharding and providing invisible automatic failover, with median read latency of 8.7 ms and single-site commit latency of 72.3 ms as perceived by the application.**

**Background.** F1 is a rewrite of Google's advertising backend, originally on manually sharded MySQL. The uncompressed dataset is tens of terabytes — small compared to many NoSQL instances but large enough to cause severe operational difficulties with manual sharding. The MySQL sharding scheme assigned each customer and all related data to a fixed shard, which enabled per-customer indexing and complex queries but required sharding knowledge in application business logic. Resharding became "extremely costly" as the customer base grew — the last resharding took **over two years** of effort across dozens of teams.

**Motivations for choosing Spanner:**
- Eliminates manual resharding (Spanner automatically reshards data).
- Provides synchronous replication and automatic failover (MySQL master-slave failover was difficult and risked data loss and downtime).
- Provides strong transactional semantics (F1 required transactions across arbitrary data and consistent reads).
- F1 needed secondary indexes, which Spanner did not natively provide but which F1 was able to implement using Spanner transactions as building blocks.

**Deployment configuration.** F1 uses 2 replicas on the west coast of the US and 3 on the east coast — chosen to cope with potential major natural disasters and to align with frontend site locations. All application writes go through F1 to Spanner by default.

**Directory-fragment distribution (Table 5).** Each directory typically corresponds to a customer. The distribution of fragments per directory shows:
- The vast majority of directories (>100M) consist of only 1 fragment — reads and writes to those customers are guaranteed single-server operations.
- A small number of directories (341 with 2–4 fragments, 5336 with 5–9) are moderately fragmented.
- Only 7 directories have 100–500 fragments — all of which are tables containing F1 secondary indexes. Writes to more than a few fragments of such tables are "extremely uncommon," observed only during untuned bulk data loads as transactions.

**Operation latencies (Table 6).** Measured from F1 servers in east-coast datacenters (where Paxos leaders are preferentially placed) over 24 hours:
- **All reads:** mean 8.7 ms, standard deviation 376.4 ms, 21.5 billion operations. The large standard deviation reflects a fat tail — partially attributed to Paxos leaders being spread across two datacenters, only one of which has machines with SSDs. Additionally, the measurement includes every read in the system, and the mean and standard deviation of bytes read were roughly 1.6 KB and 119 KB respectively.
- **Single-site commit:** mean 72.3 ms, standard deviation 112.8 ms, 31.2 million operations. The large standard deviation is caused by lock conflicts — the "fat tail" of transactions waiting on locks.
- **Multi-site commit:** mean 103.0 ms, standard deviation 52.2 ms, 32.1 million operations. Notably, the standard deviation is smaller than for single-site commits — the paper does not explain this but it may reflect the fact that multi-site commits are less likely to be on hot-spot keys that cause lock contention (single-site commits include high-contention single-directory transactions, while multi-site commits are rarer and may be on less-contended data).

**Anecdotal evidence.** Spanner's automatic failover "has been nearly invisible" to the F1 team. Unplanned cluster failures in the months preceding the paper required only that the team update their database's schema to tell Spanner where to preferentially place Paxos leaders, to keep them close to where frontends moved. Spanner's timestamp semantics made it efficient for F1 to maintain in-memory data structures computed from database state: F1 maintains a logical history log of all changes (written into Spanner as part of every transaction), takes full snapshots at a timestamp to initialize in-memory structures, and then reads incremental changes to update them.

---

### Ablation Studies and Robustness Checks

- **PRM aggregation strategy (Appendix E, Figure 13).** Comparing "min," "prod," and "last" step-wise score aggregation: "last" achieves the highest accuracy (~37% at 256 samples), "min" achieves ~35%, "prod" ~27%, and ORM ~34%. The "last" method's superiority is notable because it effectively reduces the PRM to ORM-like behavior at aggregation time, yet the PRM trained with per-step Monte Carlo labels still outperforms a separately trained ORM. The authors interpret this as evidence that step-level PRM training provides beneficial representation learning — the model learns features at intermediate steps that improve its final-step prediction even when intermediate predictions are not used at aggregation time. The discrepancy from prior work (Lightman et al., 2023; Wang et al., 2023, which found "min" to be best) is attributed to the use of soft Monte Carlo labels rather than binary correctness labels.

- **PRM vs. ORM (Appendix F, Figure 14).** The PRM consistently outperforms the ORM at all sample sizes, with the gap widening at higher sample counts. At 2048 samples, PRM best-of-N weighted reaches ~40% vs. ORM's ~35% and majority voting's ~30%. This confirms that PRM training provides a genuine accuracy benefit beyond what can be achieved with outcome-level supervision alone.

- **Revision model verifier choice (Appendix J, Figure 15a).** The base-LM PRM (trained on the base model's outputs) underperforms the revision-specific ORM (trained on the revision model's outputs) when scoring revision model outputs. At 64 generations, sequential + base-LM PRM achieves ~40% vs. sequential + revision ORM at ~42%. This confirms that distribution shift between base model and revision model outputs is a practical concern — verifiers trained on one distribution do not transfer cleanly to another.

- **Revision history in verifier context (Appendix J, Figure 15b).** Including previous revisions in the ORM's input context provides a small improvement (~1–2 percentage points at 64 generations) over the no-history ablation, but both variants outperform the parallel baseline. This demonstrates that the sequential sampling benefit is not solely attributable to the verifier seeing more context — the revision model genuinely produces better answers when conditioned on its own previous attempts.

- **Majority voting for revisions (Appendix B, Figure 10).** The sequential-to-parallel ratio trends observed with verifier-based selection are replicated with majority voting: easy questions are insensitive to ratio, hard questions show an optimal intermediate ratio, and fully sequential marginally outperforms fully parallel in aggregate. This robustness check demonstrates that the benefit of sequential revisions is not an artifact of the verifier's scoring behavior — it persists even when answers are selected by simple voting.

- **ReST^EM revision model (Appendix K, Figure 16).** An attempt to further optimize the revision model using ReST^EM (Singh et al., 2024) — a reinforcement-learning-based self-improvement procedure — backfires: additional sequential revisions substantially hurt performance with this model. At 256 generations, fully sequential performance drops to ~33.5% compared to ~38.5% at the optimal ratio. The authors hypothesize that the on-policy data collection in ReST^EM exacerbates spurious correlations in revision data, causing the model to fail to learn the revision task properly. This is a significant negative result: it demonstrates that the benefits of revision training are sensitive to the training methodology, and the successful approach (offline data construction with edit-distance-based incorrect-to-correct pairing) is not trivially replaceable by more aggressive RL-style optimization.

- **Oracle vs. predicted difficulty bins (Figures 4, 8, Appendix C, Figures 11–12).** Both oracle and predicted bins yield qualitatively similar trends across difficulty levels. In the search setting (Figure 4), predicted bins track oracle bins closely, with the curves "largely overlapping." In the revision setting (Figure 8), predicted bins show slightly lower performance at high budgets (~41% vs. ~44% at 256 generations) but still substantially outperform the parallel baseline. Appendix C (Figures 11–12) confirms that the predicted bins produce similar per-bin strategy selections. This is the critical robustness check for the compute-optimal framework: the method works without ground-truth labels.

- **Lookahead search (Figure 3, left).** Lookahead search with both k = 1 and k = 3 underperforms simpler methods (beam search, best-of-N) at the same generation budget. This is a negative result that informs the compute-optimal policy — the most powerful optimizer (in terms of verifier exploitation) produces the worst overall outcomes, consistent with the over-optimization interpretation. The extra generations spent on lookahead would have been better spent exploring more beams.

- **Beam width sweep.** The paper sweeps two beam width settings: M = √N (growing with budget) and M = 4 (fixed). The results show that the fixed M = 4 generally performs better, and this is the version used in the difficulty-bin analysis (Section 5.2, Figure 3). The paper does not provide a systematic ablation justifying why M = 4 is optimal, but the consistent use of this configuration suggests it was determined empirically.

---

### Critical Assessment

#### Does the evidence support the paper's central claims?

**Claim 1: "Spanner is the first system to distribute data at global scale and support externally-consistent distributed transactions."**

The evidence for this claim is **strong but definitional.** The paper demonstrates external consistency through a formal proof (Section 4.1.2) that depends on TrueTime's clock uncertainty bounds, and the TrueTime evaluation (Section 5.3, Figure 6) shows that ε is typically < 10 ms with rare excursions. The F1 production experience (Section 5.4) provides operational validation — no data-loss events or consistency violations are reported during the months of production use. However, the paper does not provide direct empirical validation of the external-consistency guarantee in production (e.g., a verification tool that continuously checks that commit timestamps respect real-time ordering), which would be a stronger form of evidence. The claim's strength rests on the correctness of the protocol design (which is argued formally) and the reliability of the TrueTime implementation (which is supported by measurement data), but there is no empirical demonstration that the guarantee has never been violated. This is a difficult property to verify empirically, but the absence of such verification means the claim is supported by design correctness plus operational anecdata rather than by direct measurement.

**Claim 2: "The TrueTime API and its implementation are critical to supporting external consistency and a variety of powerful features."**

**Strongly supported by architectural necessity.** The paper's design makes TrueTime the single point of dependence for external consistency, lock-free read-only transactions, and atomic schema changes. If TrueTime's guarantees were violated (if ε exceeded the assumed bound), external consistency would break. The paper demonstrates this dependence architecturally: commit wait relies on `TT.after()`, safe time relies on monotonic timestamp assignment that depends on leader-lease disjointness (which depends on TrueTime), and schema changes rely on `TT.after()` for synchronization. The TrueTime measurements (Figure 6) show that ε is well-controlled in practice, and the 5 ms commit-wait cost (Table 3) demonstrates that the approach is practical. However, the paper does not experimentally demonstrate what happens when TrueTime *fails* — e.g., by artificially inflating ε and measuring the impact on correctness or performance. Such an adversarial experiment would more forcefully demonstrate the "critical" nature of the dependency, but it was likely considered too risky for a production system.

**Claim 3: "Commit wait adds roughly 5 ms of latency, and Paxos replication keeps write latency approximately constant across 1–5 replicas."**

**Supported, with caveats about experimental setup.** Table 3 shows 14.4 ms write latency with 1 replica, 13.9 ms with 3 replicas, and 14.4 ms with 5 replicas — essentially constant. The difference between 1-replica with commit wait (14.4 ms) and 1-replica without commit wait (9.4 ms) is ~5 ms, attributed to commit wait. However, the experimental setup uses datacenters with <1 ms network distance (Section 5.1), which is not representative of the cross-continental deployment that Spanner is designed for. In a WAN setting with tens of milliseconds of latency, the relative cost of commit wait (~5 ms) would be smaller as a fraction of total latency, and the assumption that commit wait is "overlapped with Paxos communication" may hold even more strongly (since Paxos rounds take longer). The paper would be strengthened by latency measurements in a realistic multi-continent deployment, but these are not provided.

**Claim 4: "Automatic leader failover restores throughput within ~10 seconds after an unplanned datacenter failure."**

**Supported.** Figure 5 shows throughput recovery approximately 10 seconds after a hard leader kill, consistent with the 10-second default leader lease duration. The experiment clearly distinguishes between soft kill (with handoff, minimal impact), hard kill (with recovery after lease expiration), and non-leader kill (no impact). However, the paper does not explore the effect of varying lease duration on the recovery time or the steady-state overhead, which would provide guidance for practitioners tuning this parameter. The mechanism to cause slaves to release leader leases upon leader failure — mentioned as under development — would likely reduce the recovery time below 10 seconds, but this is not demonstrated.

**Claim 5: "Test-time compute with a smaller model can outperform a ~14× larger model."**

This claim is from the FLOPs-matched comparison in Section 7, and the evidence **supports it with sharp, well-characterized conditions.** The claim holds strongly for easy-to-medium problems at R ≪ 1 (e.g., +27.8% relative advantage on medium-difficulty questions with revisions) but progressively weakens as R increases or difficulty rises. On hard problems at R ≫ 1, test-time compute performs dramatically worse than the larger model (−52.9% with PRM search). The paper is transparent about these boundaries (Section 7, layered panel), which strengthens credibility.

**Genuine weaknesses in the FLOPs-matched comparison:**

1. **The ~14× larger model uses greedy decoding only.** It receives zero test-time compute augmentation — no best-of-N, no majority voting, no search. A fairer comparison would give the larger model a test-time compute budget proportional to its per-token inference cost. The current setup conflates "test-time compute vs. pretraining" with "test-time compute vs. no test-time compute," and the ~14× model's underperformance may partly reflect its lack of any inference-time strategy rather than a fundamental inferiority of pretraining.

2. **The ~14× model is parameter-scaled, not compute-optimally trained.** The paper scales only model parameters while holding training data fixed (following the LLaMA paradigm), which departs from Chinchilla-optimal scaling (where both parameters and data are scaled equally). A compute-optimally trained larger model would likely be a stronger baseline, potentially reducing or reversing the reported advantages of test-time compute. The paper acknowledges this explicitly (Section 7) but does not explore the Chinchilla-optimal comparison.

3. **Three R values are tested — a sparse sampling of a continuous space.** The transition from "test-time compute wins" to "pretraining wins" as R increases is likely smooth, but only three points are shown. The paper does not provide guidance on the critical R threshold at which pretraining becomes preferable for each difficulty level.

**Claim 6: "The optimal test-time strategy depends critically on prompt difficulty."**

**Very strongly supported** — this is the most robust finding, replicated across search (Figure 3, right) and revisions (Figure 7, right). The difficulty-dependent behavior is qualitatively different across methods: beam search hurts easy problems but helps medium ones; sequential revisions help easy problems but a balanced ratio is optimal for hard ones. The fact that these patterns are consistent across oracle and predicted difficulty bins (Figures 4, 8) and across verifier-based and majority-based selection (Figure 6, right; Appendix B, Figure 10) demonstrates that the finding is not an artifact of the difficulty estimation method or the selection mechanism.

**Genuine weaknesses in the experimental design:**

1. **Single benchmark, single model family.** All results are on the MATH benchmark (500 test questions) with PaLM 2-S\*. The paper argues the model is "representative" (Section 4), but this is an assertion, not a demonstrated fact. The results might not transfer to other model families (GPT, LLaMA, Claude), other reasoning domains (code generation, logical reasoning, scientific QA), or tasks requiring factual knowledge rather than inference. The difficulty-dependent patterns might be specific to MATH's problem distribution or PaLM 2-S\*'s error characteristics.

2. **Difficulty estimation cost is unaccounted for.** The method requires generating 2048 samples per question and scoring them with the PRM to estimate difficulty. This cost — 2048 generations per question — is not included in any budget calculation. For a single question, difficulty estimation consumes 4–8× more compute than the largest test-time budgets studied (256–512 generations). In a deployment where questions are asked once, this makes the approach wildly impractical. In a deployment where the same difficulty estimates are reused across many queries to the same questions, the amortized cost could be acceptable, but this scenario is not evaluated. The paper explicitly flags this (Section 3.2) as an open problem, acknowledging the gap.

3. **Test set of 500 questions split into quintiles of ~100 each.** With two-fold cross-validation, strategy selection is based on ~50 questions per bin per fold — a small sample that could introduce variance in the computed-optimal policy. The paper does not report confidence intervals on the compute-optimal scaling curves, making it difficult to assess statistical reliability.

4. **No combination of PRM search with revisions.** The paper studies these two mechanisms independently but never combines PRM tree-search with the revision model as the proposal distribution — a natural next step that the authors acknowledge in Section 8. The current results represent a lower bound on what combined approaches could achieve, and the omission makes it impossible to determine whether the two mechanisms provide additive or overlapping benefits.

5. **The compute-optimal policy is static — no dynamic adaptation mid-computation.** The policy selects a strategy based on pre-estimated difficulty and a fixed budget, but does not adapt if early results suggest the initial difficulty estimate was wrong or if the chosen strategy is underperforming. A dynamic policy that reallocates budget based on intermediate outcomes could potentially outperform the static policy.

6. **No latency or wall-clock analysis for revisions.** Sequential revisions are inherently serial — each revision depends on the previous one. The paper measures compute in generations but does not discuss the wall-clock implications of serial vs. parallel execution. A strategy allocating 64 sequential × 2 parallel generations takes ~32× longer wall-clock time than one using 128 parallel samples, which may be unacceptable for latency-sensitive applications regardless of accuracy gains. This tradeoff is not explored.

## 6. Limitations and Trade-offs

### 6.1 Difficulty Estimation Cost Is Unaccounted For and Prohibitive in Deployment

**The assumption or constraint.** The entire compute-optimal framework rests on the ability to estimate prompt difficulty before deciding how to allocate the inference budget. The paper's method for doing so — generating 2048 samples per question and averaging either ground-truth correctness (oracle bins) or PRM final-answer scores (predicted bins) — is extraordinarily expensive. The authors explicitly acknowledge this gap (Section 3.2):

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

**The consequence.** In a realistic deployment, the total cost of using compute-optimal test-time scaling is: `cost(difficulty_estimation) + cost(strategy_execution)`. For a single question, difficulty estimation requires 2048 generations — **4× to 8× more compute than the largest test-time budgets studied** (256–512 generations). This means the reported 4× efficiency gains over best-of-N (e.g., compute-optimal at 16 generations matching best-of-N at 64, Figure 4) are computed *after* difficulty is known, without amortizing the cost of learning it. If amortization across many queries to the same question is not possible (e.g., unique user queries, one-off analysis), the total cost is dominated by difficulty estimation, and the headline efficiency gains are unrealized. The paper's 4× figure should therefore be understood as an **upper bound on achievable efficiency**, not a realized deployment gain.

**What evidence exists in the paper.** The paper provides no experiment where difficulty-estimation cost is included in the total budget. Figures 4 and 8 report compute-optimal scaling curves where the x-axis (generation budget) reflects only strategy-execution cost. Section 3.2 flags the issue as an exploration-exploitation tradeoff but does not quantify it. The difficulty-estimation protocol (2048 samples, scored by PRM, binned into quintiles) is described in Section 3.2, and its cost can be computed from that description.

**Mitigation status.** The paper identifies this as an open problem and suggests future work on "pretraining or finetuning models to directly predict difficulty of a question" (Section 8). No such model is developed or evaluated. An adaptive approach — start with a few samples, estimate difficulty from initial scores, allocate remaining budget — is mentioned in spirit but not explored. As of the paper, the limitation is **unresolved**.

---

### 6.2 Hard Problems Remain Fundamentally Unsolved — Test-Time Compute Cannot Create Capability

**The bound.** Across all methods — PRM search (Section 5), iterative revisions (Section 6), and their compute-optimal combinations — the hardest questions (difficulty bin 5, where the base model's pass@1 is near zero) show **near-zero improvement regardless of compute budget**. In Figure 3 (right), bin 5 accuracy hovers at 1–3% for all methods and all budgets up to 256 generations. In Figure 7 (right), bin 5 shows roughly 2–3% accuracy irrespective of the sequential-to-parallel ratio. In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line is essentially flat near 0–5% — test-time compute provides essentially zero benefit, and the ~14× larger model substantially outperforms regardless of R.

**The consequence.** Test-time compute can amplify existing capability — it helps the model find correct solutions that it could have produced but might not have sampled — but it **cannot create capability from nothing**. If the base model's pass@1 is near zero on a problem class, there are no correct solutions in the proposal distribution to find (via search) or refine (via revisions). This means the approach offers **no path forward for genuinely novel or out-of-distribution reasoning** that exceeds the base model's training distribution. For such problems, scaling pretraining remains the only viable path — larger models acquire qualitatively different reasoning abilities that smaller models lack entirely, and no amount of inference-time computation compensates. This is the paper's most important boundary condition.

**What evidence exists in the paper.** Figure 3 (right, bin 5), Figure 7 (right, bin 5), Figure 9 (bin 5 line), and the FLOPs-matched bar charts in Figure 1 all show the same pattern: no method improves bin 5 performance. The paper is explicit about this finding (Section 7):

> "on the hardest problems (bin 5), test-time compute provides essentially zero benefit regardless of budget, meaning that some capabilities can only be acquired through pretraining, not recovered at inference time."

**Mitigation status.** Not mitigated and likely fundamental. The paper does not propose any mechanism for test-time compute to solve problems outside the base model's capability range. The limitation is inherent to the proposer-verifier framework: if the proposal distribution has zero probability mass on correct answers, no verifier or search algorithm can recover. The paper's contribution is precisely characterizing *where* this boundary lies (it depends on the base model, not just the dataset), not transcending it.

---

### 6.3 Single Benchmark, Single Model Family — Generality Is Unproven

**The constraint.** All experiments use the MATH benchmark (500 test questions) with a single base model family, PaLM 2-S\* (Section 4). The authors state they "believe this model is representative of the capabilities of many contemporary LLMs" (Section 4), but this assertion is not verified. Several aspects of the findings could be model- or benchmark-specific:

- **PRM quality and over-optimization behavior** depend on PaLM 2-S\*'s output distribution — a model with different calibration properties or error patterns might exhibit different difficulty-dependent scaling curves and different thresholds for verifier over-optimization.
- **Revision model effectiveness** depends on the base model's in-context learning capabilities, which vary substantially across model families — some architectures may learn revision skills more or less effectively from the edit-distance-based training procedure.
- **MATH** consists exclusively of competition-level math problems requiring symbolic reasoning and multi-step deduction. The difficulty-dependent patterns (beam search hurting easy problems due to PRM over-optimization, revisions helping easy problems, no method helping the hardest problems) may not generalize to other reasoning domains (code generation, logical reasoning, scientific QA) or to tasks requiring factual knowledge rather than deductive inference, where the nature of model errors and the utility of search/revision may differ qualitatively.

**The consequence.** Without replication on other benchmarks and model families, a practitioner cannot know whether the compute-optimal strategies learned for PaLM 2-S\* on MATH would transfer to their model or task. They would need to re-run the full analysis pipeline (2048 samples per question for difficulty estimation, strategy sweep, cross-validation) for their specific setting — an expensive proposition that undermines the practical utility of the paper's findings.

**What evidence exists in the paper.** None — this is an absence of evidence. The paper does not present results on any benchmark other than MATH, nor any model other than PaLM 2-S\* (except the ~14× larger variant used only in the FLOPs-matched comparison). Section 4 justifies the choice of MATH as a domain where test-time compute is expected to help (the model has the necessary knowledge; the challenge is complex inference), but this is a motivation for the benchmark choice, not evidence of generalizability.

**Mitigation status.** Not addressed. The paper makes no claims of generality beyond what the experiments demonstrate, but the framing of the findings as fundamental properties of test-time compute (rather than properties of PaLM 2-S\* on MATH) implies broader applicability that is empirically unsupported. Future work on other domains, benchmarks, and model families is a natural next step that the paper does not explicitly call out but that the community would need to establish generalizability.

---

### 6.4 The ~14× Larger Model Baseline Is Weak — The Pretraining-Versus-Inference Tradeoff Is Not Settled

**The constraint.** The FLOPs-matched comparison in Section 7 makes two design choices that weaken the pretraining baseline:

1. **Parameter-only scaling, not compute-optimal pretraining.** The paper scales model parameters by ~14× while holding training data fixed, following the LLaMA paradigm rather than Chinchilla-optimal scaling (where data and parameters are scaled equally per Hoffmann et al., 2022). The paper acknowledges this (Section 7):

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

A Chinchilla-optimal model trained with ~14× more total FLOPs would likely outperform a parameter-only-scaled model, making the pretraining baseline stronger than what was tested.

2. **Greedy decoding with no test-time augmentation.** The ~14× larger model uses only greedy decoding — no majority voting, no best-of-N, no search. This means the comparison is not "test-time compute vs. pretraining" but rather "test-time compute vs. no test-time compute," conflating the benefit of *any* inference-time strategy with the benefit of scaling pretraining. A fairer comparison would give the larger model a test-time compute budget proportional to its per-token inference cost — even a modest best-of-4 or best-of-8 would produce a much stronger pretraining baseline.

**The consequence.** The reported advantages of test-time compute over pretraining — e.g., +27.8% relative improvement on medium-difficulty questions with revisions at R ≪ 1 (Figure 1, top-right bar chart) — may shrink substantially or even reverse against a properly compute-optimal larger model with even rudimentary test-time augmentation. The paper's central finding about the pretraining-inference tradeoff establishes that the tradeoff *exists* and is difficulty-dependent, but the **quantitative crossover point** (at what R and difficulty level does pretraining win?) is likely misestimated in favor of test-time compute. Practitioners making budget-allocation decisions based on these numbers would be making decisions from a biased comparison.

**What evidence exists in the paper.** The bar charts in Figure 1 and the line plots in Figure 9 show the comparison as performed, with the weaknesses above. The paper's acknowledgment of the compute-optimal pretraining gap (Section 7) is explicit but does not quantify how much it matters. No sensitivity analysis explores how the crossover point would shift with a stronger baseline.

**Mitigation status.** Partially mitigated by transparency — the paper states the assumptions clearly, and the qualitative finding (test-time compute is relatively more valuable on easy problems, pretraining is necessary for hard problems, the tradeoff depends on R) is likely robust even if the quantitative crossover point shifts. However, the specific percentage advantages reported should be treated as upper bounds on test-time compute's benefit. The paper explicitly delegates the Chinchilla-optimal comparison to future work.

---

### 6.5 Sequential Revisions and Search Are Not Combined — The Full Potential of the Framework Is Unexplored

**The constraint.** The paper studies two complementary axes — PRM search (modifying how outputs are selected via verifier-guided search) and iterative revisions (modifying the proposal distribution by conditioning on previous attempts) — but **never combines them**. Section 8 explicitly acknowledges:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

**The consequence.** The two mechanisms have complementary, difficulty-dependent strengths: revisions are most effective on easy problems (where sequential refinement improves roughly-correct answers, Figure 7 right), while PRM search is most effective on medium problems (where the verifier can guide exploration toward correct solutions the model wouldn't find by random sampling, Figure 3 right). Combining them — using the revision model as the proposal distribution within beam search, or using the PRM to guide which revision branches to pursue — could yield gains beyond either method alone. The current results therefore represent a **lower bound** on what a fully integrated system could achieve, and the paper's headline 4× efficiency gains may understate what is possible.

Furthermore, the paper's finding that the ReST^EM revision model *degrades* with sequential revisions (Appendix K, Figure 16) suggests that combining the two mechanisms is non-trivial — the revision model's output distribution may not mesh cleanly with PRM-guided search, and naïve combination could produce worse results than either method alone. The absence of combination experiments leaves this central question unanswered: do revisions and search provide additive benefits, or do they interfere?

**What evidence exists in the paper.** None directly — the experiments study search (Section 5) and revisions (Section 6) in separate pipelines. The difficulty-bin analyses for each (Figure 3 right, Figure 7 right) show that their strengths are complementary across difficulty levels, providing suggestive evidence that combination would help. The ReST^EM failure (Appendix K) provides suggestive evidence that combination is not straightforward.

**Mitigation status.** Acknowledged as a direction for future work (Section 8) but not attempted. The paper provides a framework (the proposer-verifier decomposition in Section 2) that naturally accommodates the combination, and the difficulty-dependent analyses show *where* each mechanism is individually strongest, providing a roadmap for which combinations to prioritize.

## 7. Implications and Future Directions

### How This Work Changes the Landscape

This paper fundamentally shifts the conversation around distributed systems design from "clocks are unreliable, so avoid depending on them for correctness" to "clock error can be bounded tightly enough to serve as a correctness primitive, and exposing that bound through an API enables protocols that were previously dismissed as infeasible." This is a **conceptual reframing** with substantial practical consequences, though it stops short of a full paradigm shift because the enabling infrastructure (GPS + atomic clocks, tightly managed datacenter environments) limits immediate applicability outside large-scale controlled deployments.

The magnitude of the shift is best understood by examining what the paper makes newly thinkable. Before Spanner, the distributed systems community broadly accepted that external consistency (linearizability) at global scale required either a centralized timestamp oracle — which introduces an unacceptable single point of contention and failure for a planet-scale system — or was simply infeasible, necessitating weaker consistency models like eventual consistency or causal consistency. The paper's key intellectual move is to observe that a **centralized oracle is not the only way to assign globally-meaningful timestamps**: if you can bound clock uncertainty tightly enough, every node can independently assign timestamps that respect real-time ordering, with only a small wait (~`2ε ≈ 8 ms`) to guarantee correctness. This transforms external consistency from a centralized-coordination problem into a local-clock-bounding problem. The work required shifts from building a scalable timestamp service (which gets harder as the system grows) to deploying and maintaining a time-synchronization infrastructure (which is independent of database scale).

The paper resolves a long-standing tension in the systems literature between two competing narratives:

- **The "transactions don't scale" narrative** (Helland, 2007; the NoSQL movement). This position argued that general-purpose distributed transactions are fundamentally incompatible with high performance and availability at scale, so applications should be redesigned to work without them. Spanner demonstrates that this narrative was **overly pessimistic** — transactional guarantees are achievable at global scale if you invest in the right infrastructure (TrueTime, Paxos with long-lived leaders, careful engineering of the commit path). The paper takes an explicit stance: "it is better to have application programmers deal with performance problems due to overuse of transactions as bottlenecks arise, rather than always coding around the lack of transactions" (Section 2.3).

- **The "synchronized clocks are for performance, not correctness" narrative** (the Lamport logical-clock tradition). This position, deeply embedded in distributed systems pedagogy, held that physical clocks can be used for performance optimization (lease expiration, cache invalidation) but must never be relied upon for safety — correctness should depend only on logical ordering (causality, consensus). Spanner demonstrates that this narrative was **too absolute**: if clock uncertainty is bounded tightly enough, physical time can serve as a correctness primitive, and the tightness of the bound (`ε < 10 ms` in production) makes the performance cost of depending on it acceptable. This does not invalidate logical clocks — they remain essential for systems without TrueTime-grade infrastructure — but it expands the design space to include protocols that were previously considered theoretically interesting but practically infeasible.

Spanner also reconciles the gap between the **database community's** emphasis on familiar interfaces (SQL, schemas, transactions) and the **systems community's** emphasis on scalability and fault tolerance. Before Spanner, these were often treated as opposing priorities: you could have a SQL database (but it wouldn't scale globally) or a globally-distributed system (but it wouldn't support SQL or general transactions). Spanner demonstrates that the two are complementary when built on the right lower-level primitives — the semi-relational data model and SQL query language are layered on top of Paxos replication and TrueTime timestamps, and the combination is greater than the sum of its parts because TrueTime enables features (non-blocking reads in the past, atomic schema changes) that make the SQL interface more powerful than it would be on a single-machine database.

The research directions this work makes **more attractive** include:
- **Bounded clock uncertainty as a first-class cloud service.** If TrueTime is feasible at Google's scale, cloud providers could offer it as a primitive to their customers, enabling a new class of strongly-consistent distributed applications without requiring each application team to deploy GPS antennas and atomic clocks. This would democratize the architectural pattern that Spanner pioneered.
- **Protocols that assume bounded clock error rather than unbounded asynchrony.** The theoretical distributed systems literature has largely assumed an asynchronous model where clocks have unbounded drift. TrueTime demonstrates that a practically useful middle ground exists — the system is not perfectly synchronous, but the bound on asynchrony is small enough (milliseconds) to build protocols that are both correct and performant. This invites revisiting classical distributed algorithms (consensus, atomic broadcast, state machine replication) with a bounded-clock-error model.
- **Time-based optimization in storage systems.** The safe-time mechanism (`MinNextTS()`, fine-grained safe time) shows how time-based reasoning can improve read performance without compromising consistency. Other storage systems (key-value stores, object stores, file systems) could adopt similar mechanisms if they have access to bounded clock uncertainty.

Research directions this work makes **less attractive** include:
- **Pure eventual consistency as a universal default.** If strong consistency is achievable at global scale with acceptable performance overhead (commit wait adds only ~5 ms in a datacenter setting, Table 3, and is overlapped with Paxos communication in WAN settings), the argument that applications should accept eventual consistency because strong consistency is too expensive becomes weaker. Applications that genuinely need only eventual consistency (analytics, content serving) may still prefer it for maximum performance, but the default choice in a system that supports both should shift toward stronger guarantees.
- **Timestamp oracles as the only scalable approach to transaction ordering.** The paper demonstrates that a fully distributed timestamp-assignment scheme (each Paxos leader assigns timestamps independently, subject to leader-lease constraints and commit wait) can match the guarantees of a centralized oracle without the scalability bottleneck. Future systems that need global transaction ordering should consider the TrueTime pattern before defaulting to a centralized oracle or a consensus-based timestamp service.

### Follow-Up Research This Work Enables

**Replicating the compute-optimal test-time scaling analysis on code generation benchmarks, where correctness signals are cleaner and verifier training is more straightforward.** The paper's analysis is entirely on the MATH benchmark with PaLM 2-S\*. Code generation benchmarks (HumanEval, MBPP, APPS) have the advantage that correctness is decidable by unit tests rather than requiring a graded string match, and pass@k is a well-established metric. A replication study would determine whether the difficulty-dependent patterns observed here — beam search hurting easy problems due to PRM over-optimization, revisions helping easy problems, balanced sequential-parallel ratios optimal for medium difficulty — generalize to a domain where the model's error modes are different (syntax errors, logic errors, edge-case handling) and where the verifier (unit tests) has different failure characteristics than a learned PRM. A strong follow-up would sweep the same strategies (best-of-N, beam search with M=4, lookahead, sequential revisions with varying ratios) on at least two model families (e.g., CodeLlama and GPT-4) and compare the difficulty-dependent scaling curves to the MATH results. If the qualitative patterns replicate, the compute-optimal framework would be established as domain-independent; if they differ, the boundary conditions (which domains benefit from which strategies) would be empirically mapped.

**Training a lightweight difficulty estimator that predicts question difficulty from the prompt text alone, eliminating the 2048-sample cost that currently makes the approach impractical.** The paper's most significant deployment barrier is the cost of difficulty estimation — 2048 samples per question, which consumes more compute than the largest test-time budgets studied. The paper explicitly calls for "pretraining or finetuning models to directly predict difficulty of a question" (Section 8). A concrete experiment: fine-tune a small classifier (e.g., a few hundred million parameters) on (question_text, difficulty_bin) pairs generated by the full 2048-sample procedure for the 12,000 MATH training questions, then evaluate on the 500 test questions. The metrics of interest are (1) agreement between the lightweight estimator's bin assignment and the oracle (2048-sample) bin assignment, and (2) the end-to-end accuracy of compute-optimal scaling when using lightweight-estimated bins vs. oracle bins. If the lightweight estimator achieves bin agreement comparable to the PRM-based predicted bins (which the paper shows "largely overlap" the oracle bins in Figure 4), the practical feasibility of the approach would be established. If agreement is substantially worse for the hardest questions (where pass@1 is near zero and the signal is weakest), that would define a boundary condition requiring alternative approaches.

**Combining PRM-guided beam search with the revision model as the proposal distribution, measuring whether the two mechanisms provide additive or overlapping benefits.** The paper studies search and revisions in isolation but explicitly notes they were never combined (Section 8). The two mechanisms have complementary difficulty-dependent strengths: revisions work best on easy problems (where the model's initial attempts are roughly correct and need refinement, Figure 7 right), while PRM search works best on medium problems (where the verifier can guide exploration toward solutions the model wouldn't find by random sampling, Figure 3 right). A combined system would use the revision model as the generator within beam search — at each step of the beam search tree, the revision model conditions on its own previous incorrect outputs as context, potentially producing higher-quality candidate steps. Alternatively, the PRM could score revision outputs and decide when a revision chain should be terminated versus continued. A concrete experiment: for each difficulty bin, compare (1) PRM search alone, (2) revisions alone, (3) revisions + PRM best-of-N weighted (PRM selects among revision chain outputs), and (4) revisions + PRM beam search (PRM guides which revision branches to expand). The key question is whether (4) outperforms both (1) and (2) on medium-difficulty problems (bin 3, where both mechanisms individually show benefit), and whether the combination avoids the correct-to-incorrect reversion problem (38% of correct revision outputs get revised to incorrect) by using PRM scores to reject detrimental revisions. A negative result — the combination performs worse than the better of the two individual methods — would be equally informative, suggesting that the two mechanisms interfere (perhaps because the revision model's output distribution is poorly calibrated for the PRM trained on base-model outputs).

**Stress-testing verifier robustness against adversarial optimization by training PRMs with on-policy data from search and measuring the over-optimization threshold as a function of training methodology.** The paper identifies verifier over-optimization as the primary bottleneck preventing unbounded test-time compute scaling — beam search degrades easy-problem performance at high budgets (Figure 3, right), lookahead search performs worst overall (Figure 3, left) despite being the most powerful optimizer, and qualitative examples show degenerate outputs that score highly under the PRM (Appendix M). The natural follow-up is adversarial training: generate beam-search solutions (not just i.i.d. samples) and use them as additional PRM training data, with correctness labels from ground-truth answers, so the PRM learns to distinguish genuinely correct solutions from those that exploit its weaknesses. A concrete experiment: compare three PRM training regimes on the same base model — (a) i.i.d. samples only (the paper's current approach), (b) i.i.d. samples + beam search outputs at budgets 16, 64, 256, and (c) i.i.d. samples + beam search outputs + adversarial examples generated by running beam search against the PRM from (b). For each, train a PRM and measure the accuracy-vs-budget curve for beam search across the five difficulty bins. If regime (c) shows reduced or eliminated over-optimization (beam search does not degrade on easy problems at high budgets), then verifier training methodology is the key lever; if over-optimization persists even with adversarial training, then the problem is more fundamental (perhaps inherent to the PRM architecture or the Monte Carlo label generation process), suggesting that architectural innovations in verifier design are needed.

**Evaluating whether the compute-optimal policy transfers across model scales by running the full analysis on models of different sizes within the same family.** The paper uses a single base model (PaLM 2-S\*) and a single ~14× larger model for the FLOPs-matched comparison. An open question is whether the optimal strategy per difficulty bin is stable across model scales: does a 7B-parameter model benefit from the same search/revision strategy choices as a 70B-parameter model on the same problems? A concrete experiment: run the full strategy sweep (best-of-N, beam search M=4, lookahead, revisions with varying sequential-to-parallel ratios) on three model sizes within the same family (e.g., PaLM 2-XS, PaLM 2-S, PaLM 2-M) on the same MATH test set. The key measurement is the Jaccard similarity between the compute-optimal strategy maps (strategy as a function of difficulty bin and budget) across model scales. If the maps are highly similar, the strategy can be learned once on a small model and transferred to larger ones, dramatically reducing the cost of strategy discovery. If they differ — particularly if larger models have different over-optimization thresholds or different optimal revision ratios — then the strategy must be re-learned per model, which makes the approach substantially more expensive to deploy across a model family. A related question is whether difficulty bins are preserved across model scales: do models of different sizes agree on which questions are easy vs. hard? If difficulty ranking is preserved (a question that is hard for a small model is also hard for a large model, even if absolute accuracy differs), then difficulty estimation can be shared across model scales.

**Characterizing the latency-accuracy tradeoff of sequential revisions vs. parallel search, since the paper measures compute in generations but ignores wall-clock time.** The paper's compute-optimal policy on easy problems favors fully sequential revisions, and on medium problems favors a balanced sequential-to-parallel ratio. But sequential revisions are inherently serial — each revision depends on the previous one — while parallel samples can be generated simultaneously given sufficient hardware. At 64 generations, a strategy with 8 parallel chains of length 8 takes ~8× longer wall-clock time than one with 64 parallel samples, even though both consume 64 generations of compute. A concrete experiment: for a fixed wall-clock budget (not generation budget), sweep the sequential-to-parallel ratio and measure accuracy. For each ratio, the wall-clock time is determined by the longest sequential chain (since parallel chains execute concurrently), so a strategy with C parallel chains of length L has wall-clock time proportional to L (the chain length), not C × L (the generation budget). The optimal strategy under a latency constraint may be very different from the optimal strategy under a generation constraint — it may favor more parallelism even on easy problems where sequential revisions help, because the latency cost of serial execution outweighs the accuracy benefit. A follow-up would produce a Pareto frontier of (latency, accuracy) for different compute budgets and difficulty levels, giving practitioners a concrete tool for choosing their deployment configuration based on their latency requirements.

### Practical Applications and Downstream Use Cases

**On-device or edge deployment of smaller models with adaptive test-time compute for customer-support and routine-query applications.** The paper's FLOPs-matched comparison (Section 7) demonstrates that a smaller model with compute-optimal test-time strategies can match or exceed a ~14× larger model on easy-to-medium difficulty problems. For applications where the query distribution is skewed toward routine tasks (password resets, order status lookups, FAQ-style questions), deploying a small on-device or edge model with variable test-time compute is a cost-effective alternative to routing all queries to a large cloud-hosted model. The difficulty estimator serves triple duty: it allocates the test-time compute budget per query, it routes genuinely hard queries (difficulty bin 5, where test-time compute provides zero benefit) to a larger cloud model or human operator, and it avoids wasting compute on easy queries that the small model can handle with minimal augmentation. The concrete benefit is that the cloud model needs to handle only the hardest fraction of queries, which based on the MATH difficulty distribution (where bin 1 has ~80%+ accuracy and bin 5 has <5% accuracy) could be a small minority of total traffic. The hardware requirements are modest: the small model runs locally, the difficulty estimator can be a lightweight classifier (once the follow-up described above is completed), and the cloud model is an escalation path rather than the default.

**Cost-efficient data generation for LLM self-improvement and distillation pipelines.** When using LLMs to generate training data for themselves (as in STaR, ReST^EM, or rejection-sampling fine-tuning), the quality and diversity of generated solutions directly determine the quality of the resulting fine-tuned model. The paper's compute-optimal framework provides a principled way to allocate the generation budget: spend more compute on medium-difficulty problems (where search and revisions can push the model to produce correct solutions it wouldn't find by chance) and less on easy problems (where a few samples suffice) or hard problems (where no amount of compute helps, so those problems should be excluded from the training set or flagged for human annotation). The concrete workflow: for each problem in the training set, estimate difficulty using a small number of samples and the PRM's score distribution (amortizing the difficulty estimation cost across the many generations that will be done for data generation), then allocate the budget per problem according to the compute-optimal policy. The benefit is that the same total generation budget produces a higher-quality training set (more correct solutions, better coverage of solution diversity) than uniform allocation, which in turn produces a better fine-tuned model. The 4× efficiency gain reported in Figures 4 and 8 translates directly to 4× more training data for the same compute cost, or equivalent data quality at 4× lower cost.

**Batch inference pipelines for evaluation, benchmarking, and report generation where latency is irrelevant but total compute cost matters.** Organizations that run large-scale batch inference — evaluating models on benchmark suites, generating summaries of large document corpora, or producing periodic reports from structured data — care about total compute cost, not per-query latency (since queries are not user-facing and can run overnight). The compute-optimal framework applies directly: estimate difficulty once per query type (amortized across the batch), then allocate the budget per the policy. The benefit is straightforward cost savings proportional to the efficiency gain — 4× fewer GPU-hours to achieve equivalent accuracy, or higher accuracy for the same GPU-hour budget. The framework is particularly well-suited to batch settings because the difficulty estimation cost (2048 samples per query type) can be amortized across many repetitions of similar queries, making the approach cost-effective where it would not be for one-off user queries.

### When to Prefer This Method

The paper explicitly positions compute-optimal test-time scaling as an alternative to scaling pretraining compute, with the choice depending on question difficulty, the inference-to-pretraining token ratio R, and the base model's capability level. The decision rules, grounded in the FLOPs-matched comparison (Section 7) and the difficulty-bin analyses (Figures 3, 7), are:

- **Prefer compute-optimal test-time scaling with the smaller model when:**
  - The problem distribution skews toward easy-to-medium difficulty (difficulty bins 1–3, where the base model's pass@1 is non-trivially above zero and test-time compute provides meaningful gains per Figures 3 right and 7 right).
  - The inference-to-pretraining token ratio R is low (R ≪ 1, i.e., the base model is used for relatively few inference queries per training run, as in self-improvement pipelines, one-time evaluation, or low-volume high-stakes applications). In this regime, the pretraining savings from using the smaller model provide a large per-query inference budget that test-time strategies can leverage effectively.
  - Deploying the larger model is infeasible due to hardware constraints (on-device, edge deployment, memory limits) and the larger model's accuracy advantage on the problem distribution is modest.
  - A reliable verifier (PRM or ORM) can be trained on the base model's output distribution, and difficulty can be estimated (either through the amortized 2048-sample procedure or through a lightweight estimator).

- **Prefer scaling pretraining (the larger model, possibly with its own test-time augmentation) when:**
  - The problem distribution includes a substantial fraction of genuinely hard problems (difficulty bins 4–5), where the base model's pass@1 is extremely low and test-time compute provides minimal or zero improvement regardless of budget (Figures 3 right, 7 right, 9). The larger model's pretraining advantage on these problems is substantial and often insurmountable by any amount of inference-time computation — for example, the ~14× larger model outperforms compute-optimal test-time scaling by 37–53% on hard problems in the FLOPs-matched comparison (Figure 1).
  - The inference volume is very high (R ≫ 1), making per-query inference costs dominate the total compute budget. In this regime, the per-token cost advantage of the smaller model is eroded because inference FLOPs dominate pretraining FLOPs in the total budget, and the larger model's per-query accuracy advantage may outweigh its per-token cost premium.
  - Latency is critical, because the sequential revision strategies favored by the compute-optimal policy on easy problems are inherently serial and increase wall-clock time even when total FLOPs are held constant. In latency-sensitive applications (interactive assistants, real-time decision-making), the serial dependency of revisions may be unacceptable regardless of accuracy gains — a tradeoff the paper does not analyze but that practitioners must consider.
  - The base model has not been fine-tuned for revision (the revision model requires specialized training with edit-distance-based incorrect-to-correct pairing, and the benefits are sensitive to training methodology per Appendix K), and no PRM is available for the base model's output distribution. In the absence of these trained components, test-time compute options are limited to basic best-of-N or majority voting, which provide more modest gains.
