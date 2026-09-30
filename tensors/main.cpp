//! LESSSoN 1 -> simPLEST TENSOR
//easiest tensor
//will store double vals
//will have multi dimensions
//data represented using a single flat vector

#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cassert>

class Tensor{

    public:
        Tensor(
            std::vector<std::size_t> shape,
            std::vector<double> data
        ) : shape_(std::move(shape)), data_(std::move(data)){ //* std::move basically changes the ownership of the var shape to member var shape_ hence skipping ahead the lengthy copying process
            

            ////std::size_t expectedElements = 1; // *we start from 1
            std::size_t expectedElements;                                 

            //TODO : being overly strict with std::size_t max() checks 
            //* fixed by adding seperate 0 dimension check

            bool anyZeroDimensions = false;

            for(const std::size_t dimension : shape_)
            {
                if(dimension==0)
                {
                    anyZeroDimensions = true;
                    expectedElements = 0;
                    break;
                }
            }
            
            if(!anyZeroDimensions)
            {
                expectedElements = 1;
                for(const std::size_t dimension : shape_) // *if we have a scalar [] then it won't have any elements for dimension to iterate through over shape_
                {
                    //TODO : this can overflow for large shapes btw will fix later
                    //*fixed it below 

                    if(expectedElements > std::numeric_limits<std::size_t>::max()/dimension)
                    {
                        throw std::overflow_error("tensor element count overflows size_t");
                    }

                    expectedElements*=dimension; // a shape (3,4) would be having 12 elements so ... simple only
                }
            }

            //* hence expected elements after this loop for a scalar would still be 1

            if(expectedElements!=data_.size())
            {
                throw std::invalid_argument("tensor shape doesn't match it's data");
            }
        }
        //* METHODS FOR GETTING METADATA

        //!below written code isnt good bcz its asking to copy the shape_ which is inefficient
        // std::vector<std::size_t> shape()
        // {
        //     return shape_;
        // }

        //!making it by ref also is not enough as this now implies we can modify shape_
        // std::vector<std::size_t>& shape()
        // {
        //     return shape_;
        // }

        //*this is fine
        //! nodiscard tag used to warn for unused function
        //*the 1st const is like for not letting this method's caller modify the values of member var shape_ and data_ respectively that this function returns
        //*eg what if the user calls t.shape().push_back(7) this will defeat the purpose of constructor that did allat to verify that shape and data size reconcille
        //*2nd const is diff than the 1st one bcz 1st one governed what can caller do with our returned value
        //*this const controls what our function itself can do with the variables. So member var data_ and shape_ are read only here
        //*shape_.clear() won't work in here
        //* noexcept keyword basically just tells the compiler that this method would not throw exceptions whatsoever
        [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept
        {
            return shape_;
        }

        
        [[nodiscard]] const std::vector<double>& data() const noexcept
        {
            return data_;
        }

        //* no const needed here for these 2 because we are retutrning a val not a ref (notice no &)
        [[nodiscard]] std::size_t numel() const noexcept
        {
            return data_.size();
        }

        [[nodiscard]] std::size_t rank() const noexcept
        {
            return shape_.size();
        }

        //* FLATTENING a multi dimensional arrayz
        //lets try it with this example 
        //*shape = {3,2,5} -> 3 rank 
        //*flat index vector would be like 
// 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
        //eg: we need to get flat index for the element at {1,1,2}
        [[nodiscard]] double at(const std::vector<std::size_t>& idx) const //idx is the element coords in the matrix form
        {

            if(idx.size()!=rank())
            {
                throw std::invalid_argument("number of indices must match the rank of tensor");
            }

            std::size_t flatIndex = 0; //start at 0
            std::size_t stride=1; //let us start with one blocks for now

            for(std::size_t axis =0; axis<rank(); axis++)
            {
                if(idx[axis]>=shape_[axis])
                {
                    throw std::out_of_range("tensor index is outside its dimension");
                }
            }
            

            for(std::size_t axis = rank(); axis>0; axis--)
            {
                const std::size_t currentAxis = axis-1;

                ////if(idx[currentAxis]>=shape_[currentAxis])
                //{
                ////    throw std::out_of_range("tensor index is outside its dimension");
                //}

                //TODO: handle overflow
                //* the constructor invariant check guarantees that for non empty tensor
                //* the product of elements in shape would not overflow std::size_t
                //! But currently we just validate current coordinate from idx instead of validating the entirety and perform arthimetic
                //* Imagine a Tensor t({0, MAX, MAX}, {})
                //* for t.at({0, MAX-1, MAX-1}) our above check would pass 
                //! 1st loop
                // idx[2]<shape_[2]->Max-1<Max
                // flatIndex=Max-1 & stride=Max
                //! 2nd loop
                // idx[1]<shape_[1]->Max-1<Max
                // flatIndex = Max-1 + (Max-1)*Max = Max^2-1 -> already overflowed
                
                //* hence we move the check outside and validate all the coordinates at once
                flatIndex+=idx[currentAxis]*stride;
                stride*=shape_[currentAxis];
            }

            //* 1st loop 
            // flatIndex = 0 + idx[2]*1 = 0+(2*1)=2
            // stride = 1*shape[2] = 1*5=5
            //* 2nd loop
            // flatIndex = 2 + idx[1]*5 = 7 
            // stride = 5*shape[1] = 5*2=10
            //* 3rd loop
            // flatIndex = 7 + idx[0]*10 = 7+(1*10)=17
            // stride = 10*3 = 30

            return data_[flatIndex];
        }

        [[nodiscard]] std::size_t dimension(const std::size_t axis) const 
        {
            if(axis>=rank())
                throw std::out_of_range("Tensor axis is outside tensor's rank");
            
            return shape_[axis];
        }

    private:
        //! member variables
        std::vector<std::size_t>shape_;
        std::vector<double>data_;
};

int main()
{
    std::vector<std::size_t> shape_1 = {2,3};
    std::vector<double> data_1 = {0,1,2,3,4,5};
    Tensor t(
        shape_1,
        data_1
    );

    // auto shape3 = t.shape() //! this wrong (not illegal but defeats our purpose) as here shape would not be inffered as const or by ref and hence shape3 var would be a new std::vector<std::size_t> data bucket and would copy the returned value of the method to this bucket. This can be modified and what not completely defeating the puprpse
    // const auto& shape = t.shape(); //*same thing as below line no need of const as auto already propagates it
    auto& parsedShape = t.shape(); // this is alright
    // std::vector<std::size_t>& shape2 = t.shape(); //! this would be wrong because we're binding a non const ref to a const one cuz shape hands out a read only val
    auto& parsedData = t.data();

    std::size_t parsedRank = t.rank();
    std::size_t parsedNumel = t.numel();

    //* basically testing out if the constructor is working fine 
    //* i.e the shape is actually parsed by the constructor as the tensor's shape
    std::cout<<"Test 1\n";
    assert(parsedShape==shape_1);
    
    //* i.e the data we pass is actually parsed by the constructor as tensor's data
    std::cout<<"Test 2\n";
    assert(parsedData==data_1);

    //* i.e the rank of the tensor we made is calculated correctly
    std::cout<<"Test 3\n";
    //hardcoding for this test case btw
    assert(parsedRank==shape_1.size());

    //* i.e the number of elements in the tensor we made is calculated correctly
    std::cout<<"Test 4\n";
    assert(parsedNumel==data_1.size());


    //*non happy path test coverage
    {std::cout<<"Test 5\n";
    bool test5Threw = false;
    try{
        Tensor(         //* no need to wrote Tensor t2()
            {},
            {}
        );
    } catch(std::invalid_argument& error){

        assert(std::string(error.what())=="tensor shape doesn't match it's data"); // a scalar should hold a value
        test5Threw=true;
    }

    assert(test5Threw);

    std::cout<<"Test 6\n";
    //* basic scalar check
    Tensor scalar(
        {},
        {5}
    );

    assert(scalar.rank()==0);
    assert(scalar.numel()==1);
    assert(scalar.shape()==std::vector<std::size_t>{}); // the shape is indeed an empty vector as it has no dimensions so its a scalar with data = 5
    }

    {std::cout<<"Test 6\n";
    bool test6Threw = false;
    Tensor emptyMatrix(
        {1,0},
        {}
    );
    assert(emptyMatrix.rank()==2);
    assert((emptyMatrix.shape()==std::vector<std::size_t>{1,0}));
    assert(emptyMatrix.numel()==0);}

    
    { std::cout<<"Test 7\n";
    bool test7Threw = false;
    try{
        Tensor t2(
            {2,3},
            {1,2,3}
        );
    } catch(std::invalid_argument& error){

        assert(std::string(error.what())=="tensor shape doesn't match it's data");
        test7Threw=true;
    }

    assert(test7Threw);
    }   

    {std::cout<<"Test 8\n";

    //* to test the flat index fetch method
    double val = t.at({1,1});
    assert(val==4);}


    {std::cout<<"Test 9\n";
    bool test9Threw = false;
    //* to test the flat index fetch method guards#1 i.e rank mismatch
    try{
        Tensor t2(
            {2,3},
            {1,2,3,4,5,6}
        );
        double val = t2.at({2,3,4,5});
    } catch(std::invalid_argument& error){
        if(std::string(error.what())=="number of indices must match the rank of tensor")
            test9Threw = true;
    }

    assert(test9Threw);}


    {std::cout<<"Test 10\n";
    bool test10Threw = false;
    //* to test the flat index fetch method guards#2 i.e out of dimension indices
    try{
        Tensor t2(
            {2,3},
            {1,2,3,4,5,6}
        );
        double val = t2.at({4,5});
    } catch(std::out_of_range& error){
        if(std::string(error.what())=="tensor index is outside its dimension");
            test10Threw = true;
    }

    assert(test10Threw);}

    {std::cout<<"Test 11\n";
        bool test11Threw = false;
    //* to test whether size_t overflows check work
    try{
        Tensor overflow(
            {std::numeric_limits<std::size_t>::max(),3},
            {}
        );
    } catch(std::overflow_error& error){
        if(std::string(error.what())=="tensor element count overflows size_t")
            test11Threw = true;
    }
    assert(test11Threw);}

    {std::cout<<"Test 12\n";

        Tensor overflowAndZero({std::numeric_limits<std::size_t>::max(),0},{});

        assert(overflowAndZero.rank()==2);
        assert(overflowAndZero.numel()==0);
    }

    {std::cout<<"Test 13\n";

        Tensor _3dOverflow({std::numeric_limits<std::size_t>::max(),2,0},{});

        assert(_3dOverflow.rank()==3);
        assert(_3dOverflow.numel()==0);
    }

    {std::cout<<"Test 14\n";
        //* to test the dimension method
        Tensor dimTest({2,0,3},{});
        assert(dimTest.dimension(0)==2);
        assert(dimTest.dimension(1)==0);
        assert(dimTest.dimension(2)==3);

        bool rejectedAxis = false;

        try{
            auto d = dimTest.dimension(3);
        }
        catch(const std::out_of_range& error){
            if(std::string(error.what())=="Tensor axis is outside tensor's rank")
                rejectedAxis = true;
        }

        assert(rejectedAxis);
    }

    std::cout<<"Success!";

    
    return 0;
}