<img class="headerimg" src="img/17-header.jpg" alt="Photo of a thick, rusty old pipe seemingly in a basement or industrial site." />
<div class="chapter">Chapter 17</div>

# Pipeline Quality

*Machine-learning pipelines* contain the code to train, evaluate, and deploy models that are then used within products. As with all code, the code in a pipeline can and should be tested. When problems occur, code in pipelines often fails silently, simply not doing the correct thing, but without crashing. If nobody notices silent failures, problems can go undetected a long time. Problems in pipelines often result in models of lower quality. Quality assurance of pipeline code becomes particularly important when pipelines are executed regularly to update models with new data.

While machine-learned models and data quality are difficult to test, pipeline code is not really different from traditional code: it transforms data, calls various libraries, and interacts with files, databases, or networks. The conventional nature of pipeline code makes it much more amenable to the traditional software-quality-assurance approaches surveyed in chapter *[Quality Assurance Basics](14-quality-assurance-basics.md)*. In the following, we will discuss testing, code review, and static analysis for pipeline code.

## Silent Mistakes in ML Pipelines

Most data scientists can share stories of mistakes in machine-learning pipelines that remained undiscovered for a long time. These undiscovered problems were sometimes there from the beginning and sometimes occurred only later when some external changes broke something. Consider the example of a grocery delivery business with cargo bikes that, when faced with data drift, continues to collect data and regularly retrains a model to predict demand and optimal delivery routes every day. Here are some examples of possible silent mistakes:

  * At some point, the dataset is so big that it no longer fits the virtual machine used for training. Training new models fails, but the system continues to operate with the latest model, which becomes increasingly outdated. The problem is only discovered weeks later when model performance in production has degraded to the point where customers start complaining about unreliable delivery schedules.

  * The process that extracts additional training data from recent orders crashes after an update of the database connector library. Training data is hence no longer updated, but the pipeline continues to train a model every day based on exactly the same data. The problem is not observed by operators who see successful training executions and stable daily reports of the offline accuracy evaluation.

  * An external commercial weather API provides part of the data used in training the model. To account for short-term outages of the weather API, missing data is recorded with *n/a* values, which are later replaced with default values in a later data-cleaning step of the pipeline. However, when credit-card payments for the API expire unnoticed, the weather API rejects all requests—this is not noticed either for a long time because the pipeline still produces a model, albeit based on lower-quality data with default values instead of real weather data.

  * During feature engineering, time values are [encoded cyclically](https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca) to better fit the properties of the machine-learning algorithm and to better handle predictions around midnight. Unfortunately, due to a coding bug, the learning algorithm receives the original untransformed data. It still learns a model, but a weaker one than had the data been transformed as intended.

  * The system collects vast amounts of telemetry. As the system becomes popular, the telemetry server gets overloaded, dropping almost all telemetry submitted from mobile devices of delivery cyclists. Nobody notices that the amount of collected telemetry does not continue to grow with the number of cyclists and problems experienced by mobile users go undetected until users complain en masse in reviews on the app store.


A common theme here is that none of these problems manifest as a crash. We could observe these problems if we proactively monitored the right measures in the right places, but it is easy to miss them. Silent failures are typically caused by a desire for robust executions and a lack of quality assurance for infrastructure code. First, machine learning algorithms are intentionally robust to noisy data, so that they still train a model even when the data was not prepared, normalized, or updated as intended. Second, pipelines often interact with many other parts of the system, such as components collecting data, data storage components, components for data labeling, infrastructure executing training jobs, and components for deployment. Those interactions with other parts of the system are rarely well-specified or tested. Third, pipelines often consist of scripts executing multiple different steps and it may appear to work even if individual steps fail when subsequent steps can work with incomplete results or intermediate results from previous runs. Inexperienced developers often underestimate the complexity of infrastructure code. Error detection and error recovery code is often not a priority in data-science code, even when it is moved into production.

## Code Review for ML Pipelines

Data-science code can be *reviewed* just like any other code in the system. Code review can include incremental reviews of changes, but also deeper inspections of specific code fragments before deployment. As discussed in chapter *[Quality Assurance Basics](14-quality-assurance-basics.md)*, code reviews can provide many benefits at moderate costs, including discovering problems, sharing knowledge, and creating awareness in teams. For example, data scientists may discover problematic data transformations or learn tricks from other data scientists when reviewing their code or may provide suggestions for better modeling.

*During early exploratory stages,* it is usually not worth reviewing all changes to data science code in a notebook, as code is constantly changed and replaced (see also chapter *[Data Science and Software Engineering Process Models](20-data-science-and-software-engineering-process-models.md)*). However, once the pipeline is prepared for production use, it may be a good idea to review the entire pipeline code, and from there review any further changes using traditional code review. When a lot of code is migrated into production, a separate more systematic inspection of the pipeline code (e.g., an entire notebook, not just a change) could be useful to identify problems or collect suggestions for improvement. For example, a reviewer might suggest normalizing a feature before training or notice coding bugs like how the line *df["count"].astype(str).astype(int)* does not actually change any data, because it does not perform operations in place.

Code review is particularly effective for problems in data-science code that are difficult to find with testing, such as inefficient encoding of features, poor handling of data-quality issues, or poor protection of private data. Beyond data-science-specific issues, code review can also surface many other more traditional problems, including style issues and poor documentation, library misuse, inefficient coding patterns, and traditional bugs. Checklists are effective in focusing code-review activities on issues that may be otherwise hard to find.

## Testing Pipeline Components

Testing deliberately executes code with selected inputs to observe whether the code behaves as expected. Code in machine-learning pipelines that transforms data or interacts with other components in the system can be tested just like any other code.

### Testability and Modularity (“From Notebooks to Pipelines”)

It is much easier to test small and well-defined units of code than big and complex programs. Hence, software engineers typically decompose complex programs into small units (e.g., modules, objects, functions) that can each be specified and tested independently. Branching decisions like *if* statement in a program can each double the number of paths through the program, making it much more difficult to test large complex units than small and simple ones. When a piece of code has few internal decisions and limits interactions with other parts of the system, it is much easier to identify inputs to test the various expected (and invalid) executions. 

Much data-science code is initially written in notebooks and scripts, typically with minimal structure and abstractions, but with many global variables. Data-science code usually has few explicit branching decisions, but tends to be a long sequence of statements without abstractions. Data science code is also typically self-contained in that it loads data from a specific source and simply prints results, often without any parameterization. All this makes data-science code in notebooks and scripts difficult to test, because (1) we cannot easily execute the code with different inputs, (2) we cannot easily isolate and separately test different parts of the notebook independently, and (3) we may have a hard time automatically checking outputs if they are only printed to the console.

In chapter *[Automating the Pipeline](11-automating-the-pipeline.md)*, we argued to migrate pipeline code out of notebooks into modularized implementations (e.g., individual function or component per transformation or pipeline stage). Such modularization is beneficial to make the pipeline more testable. That is, the data transformation code in the middle of a notebook that modifies values in a specific data frame should be converted into a testable function that can work on different data frames and that returns the modified data frame or the result of the computation. Once modularized into a function, this code can now intentionally be tested by providing different values for the data frame and observing whether the transformations were performed correctly, even for corner cases.

<figure>

```python
# typical data science code from a notebook
df = pd.read_csv('data.csv', parse_dates=True)

# data cleaning
# ...

# feature engineering
df['month'] = pd.to_datetime(df['datetime']).dt.month
df['dayofweek']= pd.to_datetime(df['datetime']).dt.dayofweek
df['delivery_count'] = boxcox(df['delivery_count'], 0.4)
df.drop(['datetime'], axis=1, inplace=True)

dummies = pd.get_dummies(df, columns = ['month',  'weather', 'dayofweek'])
dummies = dummies.drop(['month_1', 'hour_0', 'weather_1'], axis=1)

X = dummies.drop(['delivery_count'], axis=1) 
y = pd.Series(df['delivery_count'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# training and evaluation
lr = LinearRegression()
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))
```

<figcaption>

Linear, abstraction-free data science code of how it can often be found in notebooks. This is difficult to test.

</figcaption>
</figure>

<figure>

```python
# after restructuring into separate function
def encode_day_of_week(df):
  if 'datetime' not in df.columns: raise ValueError("Column datetime missing")
  if df.datetime.dtype != 'object': raise ValueError("Invalid type for column datetime")
  df['dayofweek']= pd.to_datetime(df['datetime']).dt.day_name()
  df = pd.get_dummies(df, columns = ['dayofweek'])
  return df

# ...

def prepare_data(df):
  df = clean_data(df)
  df = encode_day_of_week(df)
  df = encode_month(df)
  df = encode_weather(df)
  df.drop(['datetime'], axis=1, inplace=True)
  return (df.drop(['delivery_count'], axis=1),
          encode_count(pd.Series(df['delivery_count'])))

def learn(X, y):
  lr = LinearRegression()
  lr.fit(X, y)
  return lr

def pipeline():
  train = pd.read_csv('train.csv', parse_dates=True)
  test = pd.read_csv('test.csv', parse_dates=True)
  X_train, y_train = prepare_data(train)
  X_test, y_test = prepare_data(test)
  model = learn(X_train, y_train)
  accuracy = eval_accuracy(model, X_test, y_test)
  return model, accuracy
```

<figcaption>

An example of the same data science code split into multiple separate functions with some error handling that can be tested independently.

</figcaption>
</figure>

All pipeline code—including data acquisition, data cleaning, feature extraction, model training, and model evaluation steps—should be written in modular, reproducible, and testable implementations, typically as individual functions with clear inputs and outputs and clear dependencies to libraries and other components in the system, if needed. 

Many infrastructure offerings for data-science pipelines now support writing pipeline steps as individual functions with the infrastructure, where the infrastructure then handles scheduling executions and moving data between functions. For example, data flow frameworks like [Luigi](https://github.com/spotify/luigi), [DVC](https://dvc.org/), [Airflow](https://airflow.apache.org/), [d6tflow](https://github.com/d6t/d6tflow), and [Ploomber](https://ploomber.io/) can be used for this orchestration of modular units, especially if steps are long-running and should be scheduled and distributed flexibly. Several cloud providers provide services to host and execute entire pipelines and experimentation infrastructure with their infrastructure, such as [DataBricks](https://databricks.com/) and [AWS SageMaker Pipelines](https://aws.amazon.com/sagemaker/pipelines/).

### Automating Tests and Continuous Integration

To test code in pipelines, we can return to standard software-testing approaches. Tests are written in a testing framework, such as *pytest*, so that test execution and reporting of results can be easily executed in an development environment—or by a continuous integration service whenever any pipeline code is modified.

<figure>

```python
def test_day_of_week_encoding():
  df = pd.DataFrame({'datetime': ['2020-01-01','2020-01-02','2020-01-08'], 'delivery_count': [1, 2, 3]})
  encoded = encode_day_of_week(df)
  assert "dayofweek_Wednesday" in encoded.columns
  assert (encoded["dayofweek_Wednesday"] == [1, 0, 1]).all()

# more tests...
```

<figcaption>

An example of a test for the ‘encode_day_of_week’ function, checking that the function correctly adds a column to the data frame with expected encoded values for the day of the week.

</figcaption>
</figure>

Similarly, it is possible to write an integration test of the entire pipeline, executing the entire pipeline with deliberate input data and checking whether the model evaluation result meets the expectation.

<figure>

```python
def test_pipeline():
  train = pd.read_csv('pipelinetest_training.csv', parse_dates=True)
  test = pd.read_csv('pipelinetest_test.csv', parse_dates=True)
  X_train, y_train = prepare_data(train)
  X_test, y_test = prepare_data(test)
  model = learn(X_train, y_train)
  accuracy = eval_accuracy(model, X_test, y_test)
  assert accuracy > 0.9
```

<figcaption>

An example of an integration test that executes multiple pipeline steps together, but on fixed test data with given accuracy expectations.

</figcaption>
</figure>

### Minimizing and Stubbing Dependencies

Small modular units of code with few external dependencies are much easier to test than larger modules with complex dependencies. For example, data cleaning and feature encoding code is much easier to test if it receives the data to be processed as an argument to the cleaning function rather than retrieving such data from an external file. For example, we can intentionally provide different inputs to function *encode_day_of_week*, which was impossible when the data source was hardcoded in the original non-modularized code.

Importantly, testing with *external* dependencies is usually not desirable if these dependencies may change between test executions or if they even depend on the live state of the production system. It is generally better to isolate the tests from such dependencies if the dependencies are not relevant to the test. For example, if the tested code sends a request to a web API to receive data, the output of the computation may change as the API returns different results, making it hard to write concrete tests. If a database is sometimes temporarily unavailable or slow, test results can appear flaky even though the pipeline code works as expected. While there is value in testing the interaction of multiple components, it is preferable to initially test behavior as much in isolation as possible to reduce complexity and avoid noise from irrelevant interference.

Not all dependencies are easy to eliminate. If moving calls to external dependencies is not feasible or desired, it is possible to replace these dependencies with a *[stub](https://en.wikipedia.org/wiki/Test_stub)* (or *[mock](https://en.wikipedia.org/wiki/Mock_object)* or *[test double](https://en.wikipedia.org/wiki/Test_double)*) during testing. A stub implements the same interface as the external dependency, but provides a simple fixed implementation for testing that always returns the same fixed result, without actually using the external dependency. More sophisticated *mock object libraries*, such as [unittest.mock](https://docs.python.org/3/library/unittest.mock.html), can help write objects with specific responses to various calls. The tests can now be executed deterministically without any external calls. 

<figure>

```python
# original implementation hardcodes external API
def clean_gender_val(df):
  def clean(row):
    if pd.isnull(row['gender']):
      row['gender'] = gender_api_client.predict(row['firstname'], row['lastname'], row['location'])
    return row
  return df.apply(clean, axis=1)

# decouple implementation from API
def clean_gender_val(df, model):
  def clean(row):
    if pd.isnull(row['gender']):
      row['gender'] = model(row['firstname'], row['lastname'], row['location'])
    return row
  return df.apply(clean, axis=1)

# test implementation with stub
def test_do_not_overwrite_gender():
  def model_stub(first, last, location):
    return 'M'

  df = pd.DataFrame({'firstname': ['John', 'Jane', 'Jim'], 'lastname': ['Doe', 'Doe', 'Doe'], 'location': ['Pittsburgh, PA', 'Rome, Italy', 'Paris, PA '], 'gender': [np.nan, 'F', np.nan]})
  out = clean_gender_val(df, model_stub)
  assert(out['gender'] ==['M', 'F', 'M']).all()
```

<figcaption>

Example data cleaning code that fills in missing gender information in our customer data. An external ML model (hosted remotely) is used to infer the gender based on the customer’s name and location. To make this code more testable, the function is decoupled from the specific API, which is now passed in as an argument. Now, we can test the cleaning code without the external API by calling the cleaning function with an alternative hardcoded implementation of the dependency (“model_stub”), which produces predictable behavior during testing. This way, multiple tests can deliberately inject different behaviors for different tests without ever having to deal with the real model inference back end.

</figcaption>
</figure>

Conceptually, *test drivers* and *stubs* are replacing the production code on both sides of the code under test. On one side, rather than calling the code from production code (e.g., from within the pipeline or from a user interface), automated unit tests act as *test drivers* that decide how to call the code under test. On the other side, *stubs* replace external dependencies (where appropriate) such that the code under test can be executed in isolation. Note that the test driver needs to set up the stub when calling the code under test, usually by passing the stub as an argument.

<figure>

![A flow chart: On the left two boxes labeled original code and test driver each point to a middle box labeled code under test, which points to two boxes on the right labeled cependencies and test stub.](./img/17-driver-stubs.svg)

<figcaption>

Test drivers execute specific tests of code units rather than the entire application, and test stubs replace dependencies during testing.

</figcaption>
</figure>

### Testing Error Handling

A good approach to avoid silent mistakes in ML pipelines is to be explicit about error handling for all pipeline code: What should happen if training data misses some values? What should happen during feature engineering if an entire column is missing? What should happen if an external model inference service used during feature engineering is timing out? What should happen if the upload of the trained model fails?

There is no single correct answer to any of these questions, but developers writing robust pipelines should consider the various error scenarios, especially regarding data quality and regarding disk and network operations. Developers can choose to (1) implement recovery mechanisms, such as filling missing values or retrying failing network connections, or to (2) signal an error by throwing an exception to be handled by the client calling the code. Monitoring how often recovery mechanisms or exceptions are triggered can help to identify when problems increase over time. Ideally, the intended error-handling behavior is documented and tested.

Both recovery mechanisms and intentional throwing of exceptions on invalid inputs or environment errors can be tested explicitly in unit tests. A unit test providing invalid inputs would either (a) assert the behavior that recovers from the invalid input or (b) assert that the code terminates with an expected exception.

<figure>

```python
def test_invalid_day_of_week_data():
  df = pd.DataFrame({'datetime_us': ['01/01/2020'], 'delivery_count': [1]})
  with pytest.raises(ValueError):
    encode_day_of_week(df) 
```

<figcaption>

An example of a unit test that ensures that the “encode_day_of_week” function correctly rejects invalid inputs (here a wrong column name) with a ValueError.

</figcaption>
</figure>

If the code has external dependencies that may produce problems in practice, it is usually a good idea to ensure that the code handles errors from those dependencies as well. For example, if data-collection code relies on a network connection that may not always be available, it is worth testing error handling for cases where the connection fails. To this end, stubs are powerful to *simulate faults* as part of a test case to ensure that the system either recovers correctly from the simulated fault or throws the right exception if recovery is impossible. Stubs can be used to simulate many different kinds of defects from external components, such as dropped network connections, slow replies, and ill-formed responses. For example, we could inject connectivity problems, behaving as if a remote server is not available on the first try, to test that the retry mechanism recovers from a short-term outage correctly, but also that it throws an exception after the third failed attempt.

<figure>

```python
## testing retry mechanism
from retry.api import retry_call
import pytest

# stub of a network connection, sometimes failing
class FailedConnection(Connection):
  remaining_failures = 0
  def __init__(self, failures):
    self.remaining_failures = failures
  def get(self, url):
    self.remaining_failures -= 1
    if self.remaining_failures >= 0:
      raise TimeoutError('fail')
    return "success"

# function to be tested, with recovery mechanism
def get_data(connection, value):
  def get(): return connection.get('https://replicate.npmjs.com/registry/'+value)
  return retry_call(get,
       exceptions = TimeoutError, tries=3, delay=0.1, backoff=2)

# 3 tests for no problem, recoverable problem, and not recoverable
def test_no_problem_case():
  connection = FailedConnection(0)
  assert get_data(connection, '') == 'success'

def test_successful_recovery():
  connection = FailedConnection(2)
  assert get_data(connection, '') == 'success'

def test_exception_if_unable_to_recover():
  connection = FailedConnection(10)
  with pytest.raises(TimeoutError):
    get_data(connection, '')
```

<figcaption>

An example of testing a recovery mechanism for a failing network connection by using a stub for that connection that deliberately injects network problems. The code should work when there are no network problems and when there are recoverable network problems, and it should throw an exception if the problem is not recoverable with three retries.

</figcaption>
</figure>

The same kind of testing for infrastructure failures should also be applied to deployment steps in the pipeline, ensuring that failed deployments are noticed and reported correctly. Again stubs can be used to test the correct handling of situations where uploads of models failed or checksums do not match after deployment.

For error handling and recovery code, it is often a good idea to log that an issue occurred, even if the system recovered from it. Monitoring systems can then raise alarms when issues occur unusually frequently. Of course, we can also write tests to observe whether the counter was correctly increased as part of testing error handling with injected faults.

<figure>

```python
from prometheus_client import Counter
connection_timeout_counter = Counter(
           'connection_retry_total',
           'Retry attempts on failed connections')

class RetryLogger():
  def warning(self, fmt, error, delay):
    connection_timeout_counter.inc()   
retry_logger = RetryLogger()

def get_data(connection, value):
  def get(): return connection.get('https://replicate.npmjs.com/registry/'+value)
  return retry_call(get,
       exceptions = TimeoutError, tries=3, delay=0.1, backoff=2,
       logger = retry_logger)
```

<figcaption>

An example using a [Prometheus](https://prometheus.io/) counter to record every time a connection fails and is retried, which can then be monitored in dashboard and alerting infrastructure like [Grafana](https://grafana.com/).

</figcaption>
</figure>

### Where to Focus Testing

Data science pipelines often contain many routine steps built with common libraries, such as *pandas* or *scikit-learn*. We usually assume that these libraries have already been tested and do not independently test whether the library’s functions are correctly implemented. For example, we would not test that scikit-learn computes the mean square error correctly, that panda’s *groupby* method is implemented correctly, or that Hadoop distributes the computation correctly across large datasets. In contrast, we should focus testing on our custom data-transformation code.

**Data quality checks.**
 Any code that receives data should check for data quality and those data quality checks should be tested to ensure that the code correctly detects and possibly repairs invalid and low-quality data. As discussed in chapter *[Data Quality](16-data-quality.md)*, data quality code typically has two parts: 

  * *Detection:* Code analyzes whether provided data meets expectations.  Data quality checks can come in many forms, including (1) checking that any data was provided at all, (2) schema validation that would detect when an API provides data in a different format, and (3) more sophisticated approaches that check for distribution stability or outliers in input data.

  * *Repair:* Code may repair data once a problem is detected. Repair may simply remove invalid data or replace invalid or missing data with default values, but repair can also take more sophisticated actions, such as imputing plausible values from context.


Code for detection and repair can both be tested with unit tests. Detection code is commonly a function that receives data and returns a boolean result indicating whether the data is valid. This can be tested with examples of valid and invalid data, as illustrated with the tests for *encode_day_of_week* in this chapter. Repair code is commonly a function that receives data and returns repaired data. For this, tests can check that provided invalid data is repaired as expected, as in the *clean_gender_val* example.

Generally, if repair for data quality problems is not possible or too many data quality problems are observed, the pipeline may also decide to raise an error. Even if repair is possible, the pipeline might report the problem to a monitoring system to observe whether the problem is common or even increasingly frequent, as illustrated with the retry mechanism in *get_data*. Both raising error messages intentionally and monitoring the frequency of repairs can avoid some of the common silent mistakes. 

**Data wrangling code.**
 Any code dealing with transforming data deserves scrutiny, especially feature-engineering code. Data transformations often need to deal with tricky corner cases, and it is easy to introduce mistakes that can be difficult to detect. Data scientists often inspect some sample data to see whether the transformed data looks correct, but rarely systematically test for corner cases and rarely deliberately decide how to handle invalid data (e.g., throw an exception or replace with default value). If data transformation code is modularized, tests can check correct transformations, check corner cases, and check that unexpected inputs are rejected.

<figure>

```python
# Variant A, returns 10 for "10k"
num = data.Size.replace(r'[kM]+$', '', regex=True).astype(float)
factor = data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
factor = factor.replace(['k','M'], [10**3, 10**6]).fillna(1)
data['Size'] = num*factor.astype(int)

# Variant B, returns 100.5000000 for "100.5M"
data["Size"] = data["Size"].replace(regex=['k'], value='000')
data["Size"] = data["Size"].replace(regex=['M'], value='000000')
data["Size"] = data["Size"].astype(str).astype(float)
```

<figcaption>

Two examples of incorrect data transformations from a Kaggle competition on Android apps that could be detected with testing. The code intends to convert textual representations of download counts (e.g., “142”, “10k”, “100M”) into numbers. Variant A but produces wrong results for some values because one the regular expressions matches the uppercase “K” instead of the used lowercase “k.” A simple test with the three numbers above could have found the bug. Variant B did not anticipate that values could contain decimal points failing on inputs like “100.5M.” Both variants fail silently, producing incorrect results for the subsequent training step. Even if decimal points were not anticipated, tests could ensure that only anticipated values were accepted, that is only combinations of numbers followed by “k” or “M,” making sure that exceptions are raised for other inputs.

</figcaption>
</figure>

**Training code.**
 Even when the full training process may ingest huge datasets and take a long time, tests with small input datasets can ensure that the training code is set up correctly. As learning is usually entirely performed by a machine-learning library, it is uncommon to test the learning code beyond the basic setup. For example, most API misuse issues and most mismatch issues of tensor dimensions and size in deep learning can be detected by executing the training code with small datasets. 

**Interactions with other components.**
 The pipeline interacts with many other components of the system, and many problems can occur there, for example, when loading training data, when interacting with a feature server, when uploading a serialized model, or when running A/B tests. These kinds of problems relate to the interaction of multiple components. It is usually worth testing that local error handling and error reporting mechanisms work as expected, as discussed earlier. Beyond that, we can test the correct interaction of multiple components with integration testing and system testing, to which we will return shortly in chapter *[System Quality](18-system-quality.md)*.

**Beyond functional correctness.**
 Beyond testing the correctness of the pipeline implementations, it can be worth considering other qualities, such as latency, throughput, and memory needs, when machine-learning pipelines operate at scale on large datasets. This can help ensure that changes to code in the pipeline do not derail resource needs for executing the pipeline. Standard libraries and profilers can be used just as in non-ML software.

## Static Analysis of ML Pipelines

While there are several recent research projects to statically analyze data science code, at the time of this writing, there are few ready-to-use tools available. As discussed in chapter *[Quality Assurance Basics](14-quality-assurance-basics.md)*, static analysis can be seen as a form of automated code review that focuses on narrow specific issues, often with heuristics. Several researchers have identified heuristics for common mistakes in data-science pipelines. If teams realize that they make certain kinds of mistakes frequently, they might be able to write a static analyzer that identifies this problem early when it occurs again.

Generic static analysis tools that are not specialized for data-science code can find common coding and style problems. For example, [Flake8](https://flake8.pycqa.org/en/latest/) is a popular static-analysis tool for Python code that can find style issues and simple bug patterns, including violations of naming conventions, misformatted documentation, unused variables, and overly complex code.

Academics have developed static-analysis tools for specific kinds of issues in data science code, and we expect to see more of them in the future. Recent examples include:

  * [Pythia](https://yanniss.github.io/tensor-ecoop20.pdf) uses static analysis to detect shape mismatch in TensorFlow programs, for example, when tensors provided to the TensorFlow library do not match in their dimensions and dimensions’ sizes.

  * [Leakage Analysis](https://github.com/malusamayo/leakage-analysis) analyzes data science code with data-flow analysis to find instances of data leakage where training and test data are not strictly separate, possibly resulting in overfitting on test data and too optimistic accuracy results.

  * [PySmell](https://arxiv.org/abs/2203.00803) and similar “code smell” detectors for data science code can detect common anti-patterns and suspicious code fragments that indicate poor coding practices, such as large amounts of poorly structured deep learning code and unwanted debugging code.

  * [mlint](https://github.com/bvobart/mllint) analyzes the infrastructure around pipeline code, statically detecting the use of “best engineering practices” such as using version control, using dependency management, and writing tests in a project.


## Process Integration and Test Maturity

As discussed in prior chapters, it is important to integrate quality assurance activities into the process. If developers do not write tests, never run their tests, never execute their static analysis tools, or just approve every code review without really looking at the code, then they are unlikely to find problems early.

Pipeline code ready for production should be considered like any other code in a system, undergoing the same (and possibly additional) quality assurance steps. It benefits from the same process integration steps as traditional code, such as, automatically executing tests with continuous integration on every commit, incremental code review, automatically reporting coverage, and surfacing static-analysis warnings during code review. 

Since data science code is often developed in an exploratory fashion in a notebook before being transformed into a more robust pipeline for production, it is not common to write tests during the early exploratory phase of a project, because much of the early code is thrown away anyway when experiments fail. This places a higher burden to write and test robust code when eventually migrating the pipeline for production. In a rush to get to market, there may be little incentive to step back and invest in testing when the data-science code in the notebook already shows promising results, yet quality assurance should probably be part of the milestone for releasing the model and should certainly be a prerequisite for automating regular runs of the pipeline to deploy model updates without developers in the loop. Neglecting quality assurances invites the kind of silent mistakes discussed throughout this chapter and can cause substantial effort to fix the system later; we will return to this dynamic in chapter *[Technical Debt](22-technical-debt.md)*.

Project managers should plan for quality assurance activities for pipelines, allocate time, and assign clear deliverables and responsibilities. Having an explicit checklist can help to assure that many common concerns are covered, not just functional correctness of certain data transformations. For example, a group at Google introduced the idea of an *[ML test score](https://research.google.com/pubs/archive/46555.pdf)*, consisting of a list of possible tests a team may perform around the pipeline, scoring a point for each of twenty-eight concerns tested by a team and a second point for each concern where tests are automated. The twenty-eight concerns include a range of different testable issues, such as whether a feature benefits the model, offline and online evaluation of models, code review of model code, unit testing of the pipeline code, adoption of canary releases, and monitoring for training-serving skew, grouped in the themes feature tests, model tests, ML infrastructure tests, and production monitoring. 

The idea of tracking the *maturity* of quality-assurance practices in a project and comparing scores across projects or teams can signal the importance of quality assurance to the teams and encourage the adoption and documentation of quality-assurance practices as part of the process. While the specific concerns from the *ML test score* paper may not generalize to all projects and may be incomplete for others, they are a great starting point to discuss what quality-assurance practices should be tracked or even required.

## Summary

The code to transform data, to train models, and to automate the entire process from data loading to model deployment in a pipeline should undergo quality assurance just as any other code in the system. In contrast to the machine-learned model itself, which requires different quality assurance strategies, pipeline code can be assured just like any other code through automated testing, code review, and static analysis. Testing is made easier by modularizing the code and minimizing dependencies. Given the exploratory nature of data science, quality assurance for pipeline code is often neglected even when transitioning from a notebook to production infrastructure; hence it is useful to make an explicit effort to integrate quality assurance into the process.

## Further Readings

  * A list of twenty-eight concerns that can be tested automatically around machine-learning pipelines and discussion of a test score to assess the maturity of a team’s quality-assurance practices: 🗎 Breck, Eric, Shanqing Cai, Eric Nielsen, Michael Salib, and D. Sculley. “[The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://research.google.com/pubs/archive/46555.pdf).” In *International Conference on Big Data (Big Data)*, 2017, pp. 1123–1132.

  * Quality assurance is prominently covered in most textbooks on software engineering, and dedicated books on testing and other quality assurance strategies exist, such as 🕮 Copeland, Lee. *[A Practitioner's Guide to Software Test Design](https://bookshop.org/books/a-practitioner-s-guide-to-software-test-design/9781580537919)*. Artech House, 2004. 🕮 Aniche, Mauricio. *[Effective Software Testing: A Developer's Guide](https://bookshop.org/books/effective-software-testing-a-developer-s-guide/9781633439931)*. Simon and Schuster, 2022; and 🕮 Roman, Adam. *[Thinking-Driven Testing](https://bookshop.org/books/thinking-driven-testing-the-most-reasonable-approach-to-quality-control/9783319731940)*. Springer, 2018.

  * Examples of academic papers using various static analyses for data science code: 🗎 Lagouvardos, Sifis, Julian Dolby, Neville Grech, Anastasios Antoniadis, and Yannis Smaragdakis. “[Static Analysis of Shape in TensorFlow Programs](https://drops.dagstuhl.de/opus/volltexte/2020/13172/).” In *Proceedings of the European Conference on Object-Oriented Programming (ECOOP)*, 2020. 🗎 Yang, Chenyang, Rachel A. Brower-Sinning, Grace A. Lewis, and Christian Kästner. “[Data Leakage in Notebooks: Static Detection and Better Processes](https://arxiv.org/abs/2209.03345).” In *Proceedings of the Int’l Conf. Automated Software Engineering,* 2022. 🗎 Head, Andrew, Fred Hohman, Titus Barik, Steven M. Drucker, and Robert DeLine. “[Managing Messes in Computational Notebooks](https://dl.acm.org/doi/abs/10.1145/3290605.3300500).” In *Proceedings of the Conference on Human Factors in Computing Systems*, 2019. 🗎 Gesi, Jiri, Siqi Liu, Jiawei Li, Iftekhar Ahmed, Nachiappan Nagappan, David Lo, Eduardo Santana de Almeida, Pavneet Singh Kochhar, and Lingfeng Bao. “[Code Smells in Machine Learning Systems](https://arxiv.org/abs/2203.00803).” arXiv preprint 2203.00803, 2022. 🗎 van Oort, Bart, Luís Cruz, Babak Loni, and Arie van Deursen. “[‘Project Smells’—Experiences in Analysing the Software Quality of ML Projects with mllint](https://arxiv.org/abs/2201.08246).” In *Proceedings of the International Conference on Software Engineering: Software Engineering in Practice,* 2022.




---
*As all chapters, this text is released under <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons BY-NC-ND 4.0</a> license.*
*Last updated on 2024-08-08.*
