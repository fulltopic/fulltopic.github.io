# CPP Notes
## Item 1: *std::unique_ptr* and *std::queue*
Define test struct:
``` c++
struct TestRefData {
	static int seq;

	int data;
	int dataSeq;

	TestRefData(int i): data(i), dataSeq(seq) {
		cout << "TestRefData constructor: " << seq << endl;
		seq ++;
	}

	TestRefData(const TestRefData& other): data(other.data), dataSeq(seq) {
		cout << "TestRefData copy constructor: " << seq << endl;
		seq ++;
	}

	TestRefData(const TestRefData&& other): data(other.data), dataSeq(seq) {
		cout << "move copy constructor: " << dataSeq << endl;
		seq ++;
	}

	TestRefData& operator= (const TestRefData& other) {
		data = other.data;
		return *this;
	}

	~TestRefData() {
		cout << "TestRefData destructor: " << dataSeq << endl;
	}
};
```
#### *std::queue*
```c++
static void testRefFunc () {
	std::queue<TestRefData> q;

	cout << "To create object " << endl;
	TestRefData d(1);

	cout << "To push " << endl;
	q.push(d);
	cout << "to get front " << endl;
	auto& dOut = q.front();
	cout << "to pop " << endl;
	q.pop();

	cout << "d: " << dOut.dataSeq << endl;

	cout << "End of test " << endl;
}
```
The output:
* Normal object construction
```bash
To create object 
TestRefData constructor: 0
```
* The *std::queue* defined *push* function as:
```c++
      void
      push(const value_type& __x)
      { c.push_back(__x); }

      void
      push(value_type&& __x)
      { c.push_back(std::move(__x)); }
``` 
So the input arguments is a reference, while the copy constructor has still been invoked. 
It is the back-end container of *std::queue* that copied the object.
So the output of *push* is:
```c++
To push 
TestRefData copy constructor: 1
```
* The return type of *front* is still a reference, so defined *dOut* as *auto&*, there is no constructor:
```c++
to get front
```
* But the after *pop* function, the object stored in back-end container of *queue* had been destructed. 
```c++
to pop 
TestRefData destructor: 1
```
* The reference *dOut* is dangling, while it is hard be detected in time
```c++
d: 1
```
* The original object destructed at the end of test
```c++
End of test 
TestRefData destructor: 0
```
To remove the dangling object, just define *dOut* as an object (not reference):
```c++
static void testRefFunc () {
	std::queue<TestRefData> q;

	cout << "To create object " << endl;
	TestRefData d(1);

	cout << "To push " << endl;
	q.push(d);
	cout << "to get front " << endl;
	auto dOut = q.front();
	cout << "to pop " << endl;
	q.pop();

	cout << "d: " << dOut.dataSeq << endl;

	cout << "End of test " << endl;
}
```
The output shows that there is another object copied:
```bash
To create object 
TestRefData constructor: 0
To push 
TestRefData copy constructor: 1
to get front 
TestRefData copy constructor: 2
to pop 
TestRefData destructor: 1
d: 2
End of test 
TestRefData destructor: 2
TestRefData destructor: 0
``` 
*std::queue* also provides *push(&&)* function:
```c++
 static void testRefFunc () {
 	std::queue<TestRefData> q;
 
 	cout << "To create object " << endl;
 	TestRefData d(1);
 
 	cout << "To push " << endl;
 	q.push(std::move(d));
 	cout << "to get front " << endl;
 	auto dOut = q.front();
 	cout << "to pop " << endl;
 	q.pop();
 
 	cout << "d: " << dOut.dataSeq << endl;
 
 	cout << "End of test " << endl;
 }
```
The output shows that only a *move* constructor invoked:
```bash
To create object 
TestRefData constructor: 0
To push 
move copy constructor: 1
to get front 
TestRefData copy constructor: 2
to pop 
TestRefData destructor: 1
d: 2
End of test 
TestRefData destructor: 2
TestRefData destructor: 0
```
#### with *std::unique_ptr*
As *std::unique_ptr* has had copy constructor deleted, following block failed to be compiled:
```c++
static void testUniquePtr () {
	std::queue<std::unique_ptr<TestRefData>> q;

	cout << "To create element " << endl;
	std::unique_ptr<TestRefData> d = std::make_unique<TestRefData>(1);

	cout << "To push " << endl;
	q.push(d); //Failed to copmile
}
```
To solve the problem, just make use of *std::move*
```c++
static void testUniquePtr () {
	std::queue<std::unique_ptr<TestRefData>> q;

	cout << "To create element " << endl;
	std::unique_ptr<TestRefData> d = std::make_unique<TestRefData>(1);

	cout << "To push " << endl;
	q.push(std::move(d));
}
```
Similarly, de-queue also requires *std::move*.

Without *std::move*, the *std::unique_ptr* would be an invalid pointer after *pop*
```c++
static void testUniquePtr () {
	std::queue<std::unique_ptr<TestRefData>> q;

	cout << "To create element " << endl;
	std::unique_ptr<TestRefData> d = std::make_unique<TestRefData>(1);

	cout << "To push " << endl;
	q.push(std::move(d));

	cout << "To get front " << endl;
	auto& dOut = q.front(); //Reference to avoid copy constructor
	if (!dOut) {
		cout << "dOut is nullptr " << endl;
	} else {
		cout << "dOut: " << dOut->dataSeq << endl;
	}

	cout << "To pop " << endl;
	q.pop();
	cout << "After pop " << endl;
	if (!dOut) {
		cout << "dOut is nullptr " << endl;
	} else {
		cout << "dOut: " << dOut->dataSeq << endl;
	}
}
```
Output:
```bash
To create element 
TestRefData constructor: 0
To push 
To get front 
dOut: 0
To pop 
TestRefData destructor: 0
After pop 
dOut is nullptr 
```
#### Workable Solution
```c++
static void testUniquePtr () {
	std::queue<std::unique_ptr<TestRefData>> q;

	cout << "To create element " << endl;
	std::unique_ptr<TestRefData> d = std::make_unique<TestRefData>(1);

	cout << "To push " << endl;
	q.push(std::move(d));

	cout << "To get front " << endl;
	auto dOut = std::move(q.front());
	if (!dOut) {
		cout << "dOut is nullptr " << endl;
	} else {
		cout << "dOut: " << dOut->dataSeq << endl;
	}

	cout << "To pop " << endl;
	q.pop();
	cout << "After pop " << endl;
	if (!dOut) {
		cout << "dOut is nullptr " << endl;
	} else {
		cout << "dOut: " << dOut->dataSeq << endl;
	}
}
```
Output:
```bash
To create element 
TestRefData constructor: 0
To push 
To get front 
dOut: 0
To pop 
After pop 
dOut: 0
TestRefData destructor: 0
```