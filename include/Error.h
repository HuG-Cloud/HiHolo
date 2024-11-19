#ifndef ERROR_H_
#define ERROR_H_    

#include <cstdlib>
#include <stdexcept>
#include <string>

// check min/max values
class BoundError: public std::logic_error
{

    public:
        BoundError(const std::string &msg): std::logic_error(msg){}
    
    
};


#endif