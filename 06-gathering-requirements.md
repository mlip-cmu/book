<img class="headerimg" src="img/06-header.jpg" alt="A photo of four folders with a lot of paper">
<div class="chapter">Chapter 6</div>

# Gathering Requirements

A common theme in software engineering is to get developers to think and plan a little before diving into coding, among others, to avoid investing effort into building solutions that do not work or do not solve the right problem. In machine learning, not different from traditional software engineering, we often are motivated by shiny technology and interesting ideas, but may fail to step back to understand the actual problem that we might solve, how users will interact with the product, and possible problems we can anticipate that may result from the product. Consequently, we may build software without much planning and end up with products that do not fit the users’ needs.

Requirements engineering invests in thinking up front about what the actual problem is, what users really need, and what complications could arise. Whether we use machine learning or not in a software system, it often turns out that user needs might be quite different from what developers initially thought they might be. Developers often make assumptions that simply will not hold in the production system. Among others, developers often underestimate how important certain qualities are for users, such as having a low response time, having an intuitive user interface, or having some feeling of agency when using the system. Some up-front investment in thinking about requirements can avoid many, often costly problems later.

After validating whether machine learning is a good match for the problem and understanding the system and user goals in chapters *[When to use Machine Learning](04-when-to-use-machine-learning.md)* and *[Setting and Measuring Goals](05-setting-and-measuring-goals.md)*, we now provide an overview of the basics of eliciting and documenting requirements. This will provide the foundation for designing ML and non-ML components to meet the needs of the system, for anticipating and mitigating mistakes, and for responsibly building systems that are safe, fair, and secure. 

Admittedly, requirements engineering often has a poor reputation among software developers—traditional approaches are commonly seen as tedious and bureaucratic, distracting from more productive-seeming coding activities. However, requirements engineering is not an all-or-nothing approach following specific rules. Even some lightweight requirements analysis in an early phase can surface insights that help improve the product. Given the challenges raised by machine learning in software systems, we believe that many developers will benefit from taking requirements engineering more seriously, at least for critical parts of the system, especially for anticipating potential problems.

<figure>

![A commit of four versions of a swing under a tree. The first panel "how the customer explained it" shows a swing with three boards for seating, the second "how the project leader understood it" shows a nonfunctional swing blocked by the tree trunk, the third "how the programmer wrote it" shows another nonfunctional swing attached to the trunk and lying on the ground, and the fourth "what the customer really needed" shows a tire swing hanging from the tree.](./img/06-req-cartoon.svg)

<figcaption>

Requirements engineering is full of challenges and misunderstandings (CC 3.0-BY [projectcartoon.com](https://web.archive.org/web/20181027030048/http://www.projectcartoon.com/)). <span title="Online-only figure; not part of the printed book.">[Online-only figure.]</span>

</figcaption>
</figure>

## Scenario: Fall Detection with a Smartwatch

For elderly people, falling can be a serious risk to their health and they may have problems getting up by themselves. In the past, *personal emergency response systems* have been deployed as devices that users could use to request help after a fall, but wearing such devices was often associated with stigma. More recently, many companies have proposed more discrete devices that detect a fall and take automated actions—including smartwatches and wall-mounted sensors. Here, we consider a software component for a smartwatch that can detect a fall based on the watch’s accelerometer and gyroscope and can contact emergency services through the connected mobile phone.

## Untangling Requirements

Software developers often feel comfortable thinking in logic, abstractions, and models, but ignore the many challenges that occur when the software *interacts with the real world*—through user interfaces, sensors, and actuators. Software is almost always built to affect the real world. Unfortunately, the real world does not always behave as software developers hope—in our fall detection scenario, software may display a message asking the user whether they are okay, but the user may just ignore it whether they are okay or not; software may recognize a falling person with machine learning but also sometimes recognizes gestures like swatting a fly as a fall; software may send a command to contact emergency responders, but nobody shows up when bad weather has interrupted phone services. 

### The World and the Machine

To untangle these concerns, it is useful to be very deliberate about which statements are about the real world and which statements are about the software, and how those relate. In software engineering, this distinction is often discussed under the label *“the World and the Machine”* after an influential paper with this title. Untangling requirements discussions with this distinction can bring a lot of clarity.

Fundamentally, software goals and user requirements are *statements about the real world*—things we want to achieve in the real world with the software. Therefore, the goals and requirements *of the system* are expressed as *desired states in the world*: for example, we might want to sell more smartwatches or help humans receive help after a fall—falling and receiving help are things happening in the real world. Software, with or without machine learning, is created to interpret parts of the world and to manipulate the world toward a desired state, either directly with sensors and actuators or indirectly mediated through human actions. 

The somewhat obvious but easily ignored problem is that *the software* itself cannot directly reason about the real world. Software takes input data, processes it, and produces output data. The input data often relates to things in the real world, but it is always necessarily mediated through sensors or humans entering data. For example, software has no direct insight into whether a human has fallen or how the smartwatch moves through physical space—instead, it has to rely on humans pressing buttons (more or less reliably) or sensors sensing movement (more or less accurately). Similarly, output data does not immediately affect the real world, but only if it is interpreted by humans who then take an action or if it controls an actuator. In our fall-detection scenario, an actuator may automatically activate a light or sound or may initiate a phone call, on which other humans then may take action to help the fallen person. Importantly, software can only reason about the world as far as real-world phenomena have been translated more or less reliably into inputs and can only affect the real world as far as output data is acted on in the intended way. 

<figure>

![A flow diagram with "the real world" and "software" as two key boxes. An arrow from the real-world to "input devices" is labeled observed phenomena and an arrow from input devices to software as "input data". Conversely an arrow from software to output devices is labeled "output data" and an arrow from output devices to the real world as "controlled phenomena".](./img/06-worldvsmachine.svg)

<figcaption>

Software can only interact with the real world mediated through input and output devices. Software cannot reason directly about the world, but only about input data that may be derived from observations about the world made by sensors. Software cannot directly influence the world, but only indirectly by presenting output data to humans or controlling phenomena in the world through actuators.

</figcaption>
</figure>

So, in order to achieve a desired system behavior in the real world, we need to consider not only how the software computes outputs from inputs, but also about what those inputs mean and how the outputs are interpreted. This usually involves various *assumptions*, such as, that the accelerometer in the smartwatch reliably captures movement of the watch in physical space or that a call to an emergency contact will bring help to the fallen person. To achieve a reliable system, it is important to critically question the assumptions and consider how to design the world beyond the software, for example, what hardware to install in a device and how to train emergency responders. Requirements engineering researcher Michael Jackson, in his original article discussing the world and machine distinction, hence draws the following conclusion: “The solutions to many development problems involve not just engineering *in* the world, but also engineering *of* the world.” That is, understanding and designing software alone is not enough, we also need to understand and design how the software interacts with the world, which may involve changes to the world outside the software.

### Requirements, Assumptions, and Specifications

Importantly, thinking clearly about the world and the machine and how the machine can only interact with the world mediated through sensors and actuators allows us to distinguish:

  * **System requirements (REQ)** describe how the *system* should operate, expressed entirely in terms of the concepts in the world. For example, the smartwatch should call emergency responders when the wearer has fallen. System requirements capture what should happen in the real world, not how software should process data.

  * **Specifications (SPEC)** describe the expected behavior of a software component in terms of input data and output data. For example, we expect a model to report “fall detected” as output if the sensor inputs are similar to previously provided training data; or the controller component should output a “call emergency responder” output 30 seconds after a “fall detected” input is received from the model, unless other input data indicates that the user has pressed a “I’m fine” button. Specifications refer only to concepts in the software world, like input and output data, but not to concepts in the real world. Specifications are sometimes also called *software requirements* or *component requirements* to contrast them from system requirements. 

  * **Assumptions (ASM)** express the relationship of real-world concepts to software inputs and outputs. For example, we assume that the gyro sensor correctly represents the smartwatch’s movement, that the GPS coordinates represent the location of the fall, that the manually entered contact address for emergency responders correctly represents the user’s intention, and that emergency responders actually respond to a call. Assumptions are what link real-world concepts in system requirements to software concepts in specifications.

  * **Implementations (IMPL)** provide the actual behavior of the software system that is supposed to align with the specification (SPEC), usually given with some code or an executable model. A detected mismatch between implementation and specification is considered a *bug*. For example, a buffer overflow in the controller crashes the system so that no “call emergency responder” output command (output) is produced if the input values representing gyro sensor readings exceed a certain value (e.g., because of an unusually hard fall).


Logically, we expect that assumptions and software specifications together should assure the required system behavior in the real world (ASM ∧ SPEC ⊨ REQ) and that the specification is implemented correctly (IMPL ⊨ SPEC). However, problems occur when:

  * The system requirements (REQ) are flat-out wrong. For example, we forget to capture that the smartwatch should not call emergency responders to the user's home if a fall is detected outside the home.

  * The assumptions (ASM) are incorrect, unrealistic, changing, or missing. For example, we (implicitly) assumed that the GPS sensor is reliable within buildings and that users always enter contact information for emergency responders correctly, but we later find that this is not always the case. As a result, the system may not meet its system requirements even when the implementation perfectly matches the specification.

  * The software specification (SPEC) is wrong. For example, the specification forgets to indicate that the “call emergency responder” output should not be produced if the input representing that user pressed a cancel button is detected. Again, the implementation may perfectly match the specification but again result in behavior that does not fulfill the system requirements.

  * Any one of these parts is internally inconsistent or inconsistent with others. For example, the software specification (SPEC) together with the assumptions (ASM) are not sufficient to guarantee the system requirements (REQ) if the specified logic when to issue calls (SPEC) does not account with a retry mechanism or redundant communication channel for possible communication issues (ASM) to ensure that emergency responders are actually contacted (REQ). 

  * The system is implemented (IMPL) incorrectly, differing from the specified behavior (SPEC), such as buffer overflows or incorrect expressions in control logic, such as actually waiting for 120 seconds rather than the specified 30 seconds.


Any of these parts can cause problems, leading to incorrect behavior in the real world (i.e., violating the system requirements). In practice, software engineers typically focus most attention on finding issues in the last category, implementation mistakes, for which there are plenty of testing tools. In practice though, incorrect assumptions seem to be a much more pressing problem in almost all discussions around safety, security, fairness, and feedback loops, with and without machine learning.

### Lufthansa 2904 Runway Crash

A classic (non-ML) example of how incorrect assumptions (ASM) can lead to catastrophe is [Lufthansa Flight 2904](https://en.wikipedia.org/wiki/Lufthansa_Flight_2904), which crashed in Warsaw in 1993 when it overran the runway after the pilot could not engage the thrust reversers in time after landing.

The airplane’s software was designed to achieve a simple safety requirement (REQ): do not engage the thrust reversers if the plane is in the air. Doing so would be extremely dangerous, hence the software should ensure that thrust reversers are only engaged to break the plane after it has touched down on the runway.

In typical world-versus-machine manner, the key issue is that the safety requirement is written in terms of real-world phenomena (“plane is in the air”), but the software simply cannot know whether the plane is in the air and has to rely on sensor inputs to make sense of the world. To that end, the Airbus software in the plane sensed the weight on the landing gear and the speed with which the plane’s wheels turn. The idea—the assumption (ASM)—was that the plane is on the ground if at least 6.3 tons of weight are on each landing gear or the wheels are turning faster than 72 knots. Both seemed pretty safe bets on how to make sense of the world in terms of available sensor inputs, providing even some redundancy to be confident in how to interpret the world’s state. The software’s specification (SPEC) was then simply to output a command to block thrust reversers unless either of those conditions hold on the sensed values.

Unfortunately, on a fatal day in 1993, due to rain and strong winds, neither condition was fulfilled for several seconds after Lufthansa flight 2904 landed in Warsaw, even though the plane was no longer in the air: the wheels did not turn fast enough due to aquaplaning, and one landing gear not was loaded with weight due to the wind. The assumption of what it means to be in the air was simply not correct for matching the status in the real world with the sensor inputs. The software hence determined based on its inputs that the plane was still in the air and thus (exactly as specified) indicated that thrust reversers must not be engaged. The real world and what the software assumed about the real world simply didn’t match. 

In summary, the system requirements (REQ) were good — the plane really should not engage thrust reversers while in the air. The implementation (IMPL) correctly matched the specification (SPEC) — the system produced the expected output for the given inputs. The problem was with the assumptions (ASM), on how the system would interpret the real world — whether it thought that the plane was in the air or not.

### Questioning Assumptions in Machine Learning

As with all other software systems, also systems with machine-learned components interact with the real world. Training data is input data derived through some sort of sensor input representing the real world (user input, logs of user actions, camera pictures, GPS locations, and so on) and predictions form outputs that are used in the real world for some manual or automated decisions. It is important to question not only the assumptions about inputs and outputs of the running system, but also assumptions made about training data used to train the machine-learning components of the system. By identifying, documenting, and checking these assumptions early, it is possible to anticipate and mitigate many potential problems. Also note that used assumptions (especially in a machine-learning context) may not necessarily be stable as the world changes. Possibly the world may even change in response to our system. Nonetheless, it may be possible to monitor assumptions over time to detect when they are still valid and require adjustment to our specification (often in terms of updating a model with new training data) or to processes outside the software as needed. 

In our fall detection scenario, we might be able to anticipate a few different problematic assumptions:

  * *Unreliable human feedback:* Users of the fall detection system may be embarrassed when falling and decide to not call for help after a fall. To that end, the user might indicate that they did not fall, which might make the system interpret the prediction as a false positive, furthermore possibly resulting in mislabeled training data for future iterations of the model. Understanding this assumption, we could, for example, clarify the user interface to distinguish between “I did not fall” and “I do not need help right now,” or be more careful about reviewing production data before using it for training.

  * *Drift:* Training data was collected from falls in one network of senior homes over an extended period. We assumed that those places are representative of the future customer base, but oversaw that users in other places (e.g., at home) may have plush carpets that result in different acceleration patterns. Checking assumptions about what training data is representative of the target distribution is a common challenge in machine-learning projects.

  * *Feedback loops:* We assume that users will universally wear the smartwatch for fall detection. However, after a series of false positives, possibly even associated with costs or stress of an unnecessary call of caretakers or emergency responders, users may be reluctant to wear the watch in certain situations, for example, when gardening. As a consequence, we may have even less reliable training data and more false positives for such situations, resulting in more users discarding their watches. We could revisit our assumptions to better understand when and why users stop wearing the watch.

  * *Attacks:* Maybe far-fetched, but malicious users could artificially shake the smartwatch on another user’s nightstand to trigger the fall detection while that user is sleeping. Depending on how worried we are about such cases, we may want to revisit our assumption that all movement of the watch happens when the watch is worn and use additional sensors to identify the user and detect that the fall happened while the watch was actually on the wrist of the right user. 


In all cases, we can question whether our assumptions are realistic and whether we should strengthen the system with additional sensor inputs or weaken the system requirements to be less ambitious and more realistic.

To emphasize this point more, let us consider a second scenario of product recommendations in an online shop, where the system requirement might be to rank those products highly that many real-world customers like:

  * *Attacks:* When building a recommendation system, we may (implicitly) assume that past reviews reflect the honest opinions of past customers and that past reviews correspond to a representative or random sample of past customers. Even just explicitly writing down such assumptions may reveal that they are likely problematic and that it is possible to deliberately influence the system with malicious tainted inputs (e.g., fake reviews, review bombing), thus violating the system requirement when highly rated products may not actually relate to products that many customers like in the real world. Reflecting on these assumptions, we can consider other design decisions in the system, such as only accepting reviews from past customers, a reputation system for valuing reviews from customers writing many reviews, or using various fraud detection mechanisms. Such alternative design decisions often rely on weaker or fewer assumptions.

  * *Drift:* We may realize that we should not assume that reviews from 2 years ago necessarily reflect customer preferences today and should correspondingly adjust our model to consider the age of reviews. Again, we can revise our assumptions and adjust the specification so that the system still meets the system requirements. 

  * *Feedback loops:* We might realize that we might introduce a feedback loop since we assume that recommendations affect purchase behavior, while recommendations are also based on purchase behavior. That is, we may shape the system such that products ranked highly early on remain ranked highly because they are bought and reviewed the most. Unintentional feedback loops often occur due to incorrect or missing assumptions that do not correctly reflect processes in the real world.


Reflecting on assumptions reveals problems and encourages revisions to the system design. In the end, we may decide that a system requirement to rank the products by average customer preferences may be unrealistic in this generality and we need to weaken requirements to more readily measurable properties or strengthen the software specification — doing this would be more honest about what the system can achieve and forces us to think about mitigations to various possible problems. In all of these cases, we critically reflect on assumptions and whether the stated requirements are actually achievable, possibly retreating to weaker and more realistic requirements or more complex specifications that avoid simple shortcuts.

### Behavioral Requirements and Quality Requirements

As a final dimension to untangle requirements, system requirements and software specifications are typically distinguished into behavioral requirements and quality requirements (also often known as  functional versus nonfunctional requirements). While the distinction can blur for some requirements, *behavioral requirements* generally describe what the software should do, whereas *quality requirements* describe how the software should do it or how it should be built. For example, in the fall-detection scenario, a behavioral system specification is *“call emergency services after a fall”* and a behavioral software specifications may include *“if receiving the ‘fall detected’ input, activate ‘call emergency responder’ output after 30 seconds;”* a quality system requirement could be *“the system should have operational costs of less than $50 per month”* and a quality software specification may be *“the fall detector should have a latency of under 100ms.”* For all quality requirements, defining clear measures to set specific expectations is important, just as for any goals (see chapter *[Setting and Measuring Goals](05-setting-and-measuring-goals.md)*). 

There are many possible quality requirements. Common are qualities like safety, security, reliability, and fairness; qualities relating to execution performance like latency, throughput, and cost; qualities relating to user interactions like usability and convenience; as well as qualities related to the development process like development cost, development duration, and maintainability and extensibility of the software. All these qualities can be discussed for the system as a whole as well as for individual software components and we may need to consider assumptions related to them. Often designers and customers focus on specific quality requirements but ignore others in early stages, but often it becomes quickly noticeable after deployment if the system misses some important qualities like usability or operational efficiency. We will return to various qualities for machine-learning components in chapter *[Quality Attributes of ML Components](09-quality-attributes-of-ml-components.md)* and to quality requirements for operations, known as *service-level objectives*, in chapter *[Planning for Operations](13-planning-for-operations.md)*.

## Eliciting Requirements



Understanding the requirements of a system is notoriously difficult. 

First, customers often provide a vague description of what they need, only to later complain *“no, not like that.”* For example, when we ask potential users of the fall detection smartwatch, they may say that they prefer to fully automate calling emergency responders, but revise that once they experience that the system sometimes mistakenly detects a hand gesture as a fall. In addition, customers tend to focus on specific problems with the status quo, rather than thinking more broadly about what a good solution would look like. For example, customers may initially focus on concealing the fall detection functionality in the smartwatch to avoid the stigma of current solutions, but not worry about any further functionality.

Second, it is difficult to capture requirements broadly and easy to ignore specific concerns. We may build a system that makes the direct customer happy, but everybody else interacting with the system complains. For example, the smartwatch design appeals to elderly users buying the watch, but it annoys emergency responders with poorly designed notifications. Especially concerns of minorities or concerns from regulators (e.g., privacy, security, fairness) can be easy to miss if they are not brought up by the customers.

Third, engineers may accept vagueness in requirements assuming that they already understand what is needed and can fill in the gaps. For example, engineers may assume that they have a good intuition by themselves of how long to wait for a confirmation of a fall before calling emergency responders, rather than investigating the issue with experts or a user study.

Fourth, engineers like to focus on technical solutions. When technically apt people, including software engineers and data scientists, think about requirements, they tend to focus immediately on software solutions and technical possibilities, biased by past solutions, rather than comprehensively understanding the broader system needs. For example, they may focus on what information is available and can be shown in a notification, rather than identifying what information is actually needed by emergency responders. 

Finally, engineers prefer elegant abstractions. It is tempting to provide general and simple solutions that ignore the messiness of the real world. For example, tremors from Parkinson's disease might make accelerometer readings problematic, but may be discarded as an inconvenient special case that does not fit into the general design of the system.

As a response to all these difficulties, the field of requirements engineering has developed several techniques and strategies to elicit requirements more systematically and comprehensively.

### Identifying Stakeholders

When eliciting goals and requirements, we often tend to start with the system owners and potential future users and ask them what they would want. However, it is important to recognize that there are potentially many people and organizations involved with different needs and preferences, not all of whom have a direct say in the design of the system. A first step in eliciting requirements hence is to identify everybody who may have concerns and needs related to the system and who might hence provide useful insights into requirements, goals, and concerns. By listening to a broad range of people potentially using the system, being affected by the system, or otherwise interested in the system, we are more likely to identify concerns and needs more broadly and avoid missing important requirements.

In software engineering and project management, the term *stakeholder* is used to refer to all persons and entities who have an interest in a project or who may be affected by the project. This notion of stakeholder is intentionally very broad. It includes the organization developing the project (e.g., the company developing and selling the fall detection device), all people involved in developing the project (e.g., managers, software engineers, data scientists, operators, companies working on outsourced components), all customers and users of the project (e.g., elderly, their caretakers, retirement homes), but also people indirectly affected positively or negatively (e.g., emergency responders, competitors, regulators concerned about liability or privacy, investors in the company selling the device). Preferences of different stakeholders rarely have equal weight when making decisions in the project, but identifying the various stakeholders and their goals is useful for understanding different concerns in the project and deliberating about trade-offs, goals, and requirements.

It is usually worthwhile to conduct a brainstorming session early in the project to list all potential stakeholders in the project. With such list, we can start thinking about how to identify their needs or concerns. In many cases, we might try to talk to representative members of each group of stakeholders, but we may also identify their concerns from indirect approaches, such as background readings or personas.

### Requirements Elicitation Techniques

Most requirements are identified by *talking* to the various stakeholders about what problems they face in the status quo and what they need from the system to be developed. 

Especially *interviews* with stakeholders are a common form to elicit requirements. We can ask stakeholders about the problems they hope to solve with the system, how they envision the solution, and what qualities they would expect from it (e.g., price, response time). Interviews can give insight into how things *really* proceed in practice and allow soliciting a broad range of ideas and preferences for solutions from a diverse set of stakeholders. We can also explicitly ask about trade-offs, such as whether a more accurate solution would be worth the additional cost. In our fall detection scenario, we could ask potential future users, users of the existing systems that we hope to replace, caregivers, emergency responders, and system operators about their preferences, ideas, concerns, or constraints.

Interviews are often more productive if they are supplemented with visuals, storyboards, or prototypes. If our solution will replace an existing system, it is worth talking about concrete problems with that system. With interactive prototypes, we can also observe how potential users interact with the envisioned system. In a machine-learning context, *[wizard-of-oz experiments](https://en.wikipedia.org/wiki/Wizard_of_Oz_experiment)* are particularly common, in which a human manually performs the task of the yet-to-be-developed machine-learning components—for example, a human operator may watch a live video of a test user to send messages to the smartwatch or call emergency responders. Wizard-of-oz experiments allow us to quickly create a prototype that people can interact with, without having to solve the machine-learning problem first.

Typically, interviews are conducted after some background study to enable interviewers to ask meaningful questions, such as learning about the organization and problem domain and studying existing systems to be replaced. We can try to reuse knowledge, such as quality trade-offs, from past designs. For example, we could read academic studies about existing fall detection and emergency response systems, read public product reviews of existing products to identify pain points, and try competing products on the market to understand their design choices and challenges. We can also read laws or consult lawyers on relevant regulations.

There are also many additional methods beyond interviews and document analysis, each specialized in different aspects. Typical approaches include:

  * *Surveys* can be used to answer more narrow questions, potentially soliciting inputs from a large population. *Group sessions* and *workshops* are also common to discuss problems and solutions with multiple members of a stakeholder group. 

  * *Personas* have been successfully used to think through problems from the perspective of a specific subgroup of users, if we do not have direct access to a member of that subgroup. A persona is a concrete description of a potential user with certain characteristics, such as a technology-averse elderly immigrant with poor English skills, that helps designers think through the solution from that user’s perspective. 

  * *Ethnographic studies* can be used to gain a detailed understanding of a problem, where it may be easier to experience the problem than have it explained: here requirements engineers either passively observe potential users in their tasks and problems, possibly asking them to verbalize while they work, or they actively participate in their task. For example, a requirements engineer for the fall detection system might observe people in a retirement home or during visits with caregivers who recommend the use of an existing personal emergency response system.

  * *Requirements taxonomies and checklists* can be used across projects to ensure that common requirements are at least considered. This can help to ensure that quality requirements about concerns like response time, energy consumption, usability, security, privacy, and fairness are considered even if they were not brought up organically in any interviews. 


### Negotiating Requirements

After an initial requirements elicitation phase, for example, conducting document analysis and interviews, the developer team has usually heard of a large number of problems that the system should solve and lots of ideas or requests for how the system should solve them. They now need to decide which specific problems to address and how to prioritize the various qualities in a solution.

Requirements engineering and design textbooks can provide some specific guidance and methods, such as *[card sorting](https://en.wikipedia.org/wiki/Card_sorting)*, *[affinity diagramming](https://en.wikipedia.org/wiki/Affinity_diagram)*, and *[importance-difficulty matrices](https://spin.atomicobject.com/2018/03/06/design-thinking-difficulty-importance-matrix/)*, but generally the goal is to group related data, identify conflicting concerns and alternative options, and eventually arrive at a decision on priorities and conflict resolution. For example, different potential users that were interviewed may have stated different views on whether emergency responder calls fully automatically after a detected fall—system designers can decide whether to pick a single option (e.g., full automation), a compromise (e.g., call after 30 seconds unless the user presses the “I’m fine” button), or leaves this as a configuration option to the user.

In many cases, conflicts arise between the preferences of different stakeholders, for example, when the wishes of users clash with those of developers and operators (e.g., frequent cheap updates versus low system complexity), when the wishes of customers clash with those of other affected parties (e.g., privacy preferences of end users versus information needs of emergency responders), or when goals of the product developers clash with societal trends or government policy goals (e.g., maximizing revenue versus stronger privacy rights and lowering medical insurance premiums). Conflicting preferences are common, and developers need to navigate conflicts and trade-offs constantly. In a few cases, laws and regulations may impose constraints that are difficult to ignore, but, in many others, system designers will have to apply engineering judgment or business judgment. The eventual decision usually cannot satisfy all stakeholders equally but necessarily needs to prioritize the preferences of some over others. The decision process can be iterative, exploring multiple design options and gathering additional feedback from the various stakeholders on these options. In the end, somebody usually is in a position of power to make the final decisions, often the person paying for the development. The final requirements can then be recorded together with a justification about why certain trade-offs were chosen.

### Documenting Requirements

Once the team settles on requirements, it is a good idea to write them down so they can be used for the rest of the development process. Requirements are typically captured in textual form as “*shall”* statements about what the system or software components shall do, what they shall not do, or what qualities they shall have. Such documents should also capture assumptions and responsibilities of non-software parts of the system. The documented requirements reflect the results of negotiating conflicting preferences.

Classic requirements documents (Software Requirements Specification, SRS) have a reputation for being lengthy and formulaic. It is not uncommon for such documents to contain hundreds of pages of nested bullet points and text, such as the following for our scenario:

  * *Behavioral Req. 3.1: The system shall prompt the user for confirmation before initiating emergency contact.*

  * *Behavioral Req. 3.2: If no user response is detected within 30 seconds, the system shall proceed with emergency contact.*

  * *Behavioral Req. 4.1: The system shall access and transmit the user's GPS location to the selected emergency contact or services when a fall is detected.*

  * *Quality Req. 4.1: The user interface shall be usable by individuals with varying levels of physical ability, including those with visual and motor impairments.*

  * *Quality Req. 7.2: The system's interface shall be simple and intuitive to ensure ease of setup and configuration by users. The interface shall minimize the complexity and number of steps required to complete the setup process and allow easy access to configuration options. Following the documentation, an average user should be able to set up the fall detection feature within 5 minutes.*

  * *Hardware Assumption 5: The smartwatch and connected mobile phone will have a consistent and stable Bluetooth connection.*

  * *User Assumption 1: The user will wear the smartwatch correctly and keep it charged as per the manufacturer's recommendations.*

  * *User Responsibility 1: The user is responsible for regularly updating their personal and emergency contact information in the system.*


Formal requirements documents can be used as the foundation for contracts when one party contracts another to build the software. Beyond that, clearly documented requirements are also useful for deriving test cases, planning and estimating work, and creating end-user documentation.

At the same time, more lightweight approaches to requirements documentation have emerged from agile practices, trying to capture only essential information and partial requirements with short notes in text files, issue trackers, or wikis. In particular user stories, requirement statements in the form *“As a [type of user], I want [an action] so that [a benefit/value],”* are common to describe how end users will interact with the system, such as:

  * *As a user with health concerns, I want the smartwatch to automatically call for emergency assistance if I'm unable to respond within 30 seconds of a fall detection, ensuring that help is dispatched in situations where I might be incapacitated.*

  * *As an active senior who enjoys walks, I want the smartwatch to send my GPS location to emergency contacts or services when a fall is detected, so that I can be quickly located and assisted outside my home, even if I'm unable to communicate.*

  * *As a user who is not tech-savvy, I want the smartwatch setup and configuration process to be simple and intuitive, so I can do it without needing assistance from others.*


Typically, more rigorous requirements documents systematically covering also quality requirements and environment assumptions become more important as the risks of the project increase.

### Requirements Evaluation

Documented requirements can be evaluated and audited, similar to code review. Requirements documents can be shown to different stakeholders in a system to ask them whether they agree with the requirements and whether they find anything missing. Often, simple prototypes based on identified requirements can elicit useful feedback, where stakeholders immediately notice issues that they did not previously think about.

Checklists can ensure that important quality requirements are covered (e.g., privacy, power efficiency, usability). In addition, we can evaluate requirements documents systematically for clarity and internal consistency. For example, we can check that measures for qualities in the document are all clearly specified and realistic to measure in practice, that system requirements and software specifications are clearly separated, that assumptions are stated, that all statements are free from ambiguity, and that the document itself is well structured and identifies the sources of all requirements.

Asking domain experts to have a look at requirements is particularly useful to check whether assumptions (ASM) are realistic and complete, for example, whether emergency responders would require a contractual agreement to respond to automated calls. Assumptions can also be validated with experiments and prototypes, such as, whether accurately detecting falls from accelerometer and gyroscope data is feasible and measuring the typical response time from an emergency team. In projects with machine-learning components, AI literacy is crucially important during the requirements engineering phase to catch unrealistic requirements (e.g., “no false positives acceptable”) and unrealistic assumptions (e.g., “training data is unbiased and representative”).

Note that all this evaluation can be done before the system is fully built. This way, many problems can be detected early in the development process, when there is still more flexibility to redesign the system.

## How Much Requirements Engineering and When?

As we will discuss in chapter *[Data Science and Software Engineering Process Models](20-data-science-and-software-engineering-process-models.md)*, different projects benefit from different processes and different degrees of formality. Establishing firm requirements up front can help to establish a contract and can reduce the number of changes later, and it can provide a strong foundation for designing a system that actually fulfills the quality requirements. However, requirements are often vague or unclear early on until significant exploration is done—especially in machine-learning projects, where experimentation is needed to understand technical capabilities. Hence, heavy early investment in requirements can be seen as an unnecessary burden that slows down the project.

In practice, requirements are rarely ever fully established up front and then remain stable. Indeed, there is often a common interaction between requirements and design and implementation. For example, as software architects identify that certain quality goals are unrealistic with the planned hardware resources or as developers try to implement some requirements and realize that some ideas are not feasible (e.g., in a prototype), the team may revisit requirements, possibly even changing or weakening the goal of the entire system. Similarly, customers and stakeholders may identify problems or new ideas only once they see the system in action, be it as an early prototype or a deployed near-final product. Importantly though, the fact that requirements may change is not a reason not to elicit or analyze requirements at all. Reasoning about requirements often helps to identify problems early and reduce changes later.

Many low-risk projects will get away with very lightweight and informal requirements engineering processes where developers incrementally think through requirements as they build the system, show minimal viable products to early users to receive feedback, and only take minimal notes—arguably all following agile development practices. However, we argue that many software projects with machine-learning components are very ambitious and have substantial potential for harm—hence we strongly recommend that more risky projects with machine-learning components take requirements engineering seriously and to clearly think through requirements, assumptions, and specifications. It may be perfectly fine to delay a deeper engagement with requirements until later in the project when transitioning from a proof-of-concept prototype to a real product. However, if wrong and biased predictions can cause harm or if malicious actors may have a motivation to attack the system, an eventual more careful engagement with requirements will help responsible engineers to anticipate and mitigate many problems as part of the system design before deploying a flawed or harmful product into the world.

## Summary 

Requirements engineering is important to think through the problem and solution early before diving into building it. Many software problems emerge from poor requirements rather than buggy implementations, including safety, fairness, and security problems—these kinds of requirements problems are difficult to fix later.

To understand requirements, it is useful to explicitly think about system requirements in the real world as distinct from software specifications of the technical solution, and the assumptions needed to assure the system requirements with the software behavior. Many problems in software systems with and without machine learning come from problematic assumptions, and investing in requirements engineering can help to identify and mitigate such problems early on.

The process for eliciting requirements is well understood and typically involves interviews with various stakeholders, followed by synthesis and negotiation of final requirements. The resulting requirements can then be documented and evaluated.

Machine learning does not change much for requirements engineering overall. We may care about additional qualities, set more ambitious goals, expect more risks, and face more unrealistic assumptions, but the general requirements engineering process remains similar. With additional ambitions and risks of machine learning, investing in requirements engineering likely becomes more important for software projects with machine-learning components than for the average software project without.

## Further Readings

  * A good textbook on requirements engineering, going into far more depth than we can do here: 🕮 Van Lamsweerde, Axel. *[Requirements Engineering: From System Goals to UML Models to Software](https://bookshop.org/books/requirements-engineering-from-system-goals-to-uml-models-to-software-specifications/9780470012703).* John Wiley & Sons, 2009.

  * A seminal software engineering paper on challenges in requirements engineering, in particular from not clearly distinguishing the phenomena of the world from those of the machine: 🗎 Jackson, Michael. “[The World and the Machine](https://web.archive.org/web/20170519054102id_/http://mcs.open.ac.uk:80/mj665/icse17kn.pdf).” In *Proceedings of the International Conference on Software Engineering*. IEEE, 1995.

  * A paper strongly arguing for the importance of requirements engineering when building software with ML components, including a discussion of the various additional qualities that should be considered, such as data quality, provenance, and monitoring: 🗎 Vogelsang, Andreas, and Markus Borg. “[Requirements Engineering for Machine Learning: Perspectives from Data Scientists](https://arxiv.org/pdf/1908.04674.pdf).” In *Proceedings of the International Workshop on Artificial Intelligence for Requirements Engineering (AIRE)*, 2019.

  * A paper outlining a path of how requirements engineering can be useful in better understanding the domain and context of a problem and how this helps in better curating a high-quality dataset for training and evaluation of a model: 🗎 Rahimi, Mona, Jin LC Guo, Sahar Kokaly, and Marsha Chechik. “[Toward Requirements Specification for Machine-Learned Components](https://ieeexplore.ieee.org/document/8933771).” In *IEEE International Requirements Engineering Conference Workshops*, pp. 241–244. IEEE, 2019.

  * A rare machine-learning paper that explicitly considers the difference between the world and the machine and how fairness needs to be considered as a system problem, not just a model problem: 🗎 Kulynych, Bogdan, Rebekah Overdorf, Carmela Troncoso, and Seda Gürses. “[POTs: Protective Optimization Technologies](https://arxiv.org/abs/1806.02711).” In *Proceedings of the Conference on Fairness, Accountability, and Transparency*, pp. 177–188. 2020.

  * A position paper arguing for the importance of requirements engineering and considering many different stakeholders when building ML-enabled products in a healthcare setting: 🗎 Wiens, Jenna, Suchi Saria, Mark Sendak, Marzyeh Ghassemi, Vincent X. Liu, Finale Doshi-Velez, Kenneth Jung et al. “[Do No Harm: A Roadmap for Responsible Machine Learning for Health Care](https://scholar.harvard.edu/files/finale/files/do_no_harm-_a_roadmap_for_responsible_machine_learning_for_healthcare.pdf).” *Nature Medicine* 25, no. 9 (2019): 1337–1340.

  * A paper illustrating the importance of engaging with requirements engineering at the system level and negotiating the preferences of diverse stakeholders when thinking about fairness in machine learning: 🗎 Bietti, Elettra. “[From Ethics Washing to Ethics Bashing: A View on Tech Ethics from within Moral Philosophy](https://dl.acm.org/doi/pdf/10.1145/3351095.3372860).” In *Proceedings of the Conference on Fairness, Accountability, and Transparency*, pp. 210–219. 2020.

  * An illustrative example of using personas to reason through requirements from the perspective of a different person: 🗎 Guizani, Mariam, Lara Letaw, Margaret Burnett, and Anita Sarma. “[Gender Inclusivity as a Quality Requirement: Practices and Pitfalls](https://par.nsf.gov/servlets/purl/10226813).” *IEEE Software* 37, no. 6 (2020).




---
*As all chapters, this text is released under <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons BY-NC-ND 4.0</a> license.*
*Last updated on 2024-06-13.*
