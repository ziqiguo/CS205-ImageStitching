template <class T>
class Vector
{
public:
    typedef T * iterator;
    
    Vector();
    Vector(unsigned int size);
    Vector(unsigned int size, const T & initial);
    Vector(const Vector<T> & v);      
    ~Vector();

    unsigned int capacity() const;
    unsigned int size() const;
    bool empty() const;
    iterator begin();
    iterator end();
    T & front();
    T & back();

    #pragma acc routine seq
    void push_back(const T & value); 
    void pop_back();  

    void reserve(unsigned int capacity);   
    void resize(unsigned int size);   

    T & at(unsigned int index); 

    T & operator[](unsigned int index);  
    Vector<T> & operator=(const Vector<T> &);
    void clear();
private:
    unsigned int my_size;
    unsigned int my_capacity;
    T * buffer;
};
