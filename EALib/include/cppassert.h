#pragma once

#include <stdexcept>

class assertion_failure : public std::exception
{
  public:
    assertion_failure(const char *message) : message(message){};
    const char *message;

    const char *what() const throw()
    {
        return message;
    }
};

#define EXPAND_MACRO_SUB(macro) #macro
#define EXPAND_MACRO(macro) EXPAND_MACRO_SUB(macro)
#define t_assert(expression, message)                                                                                  \
    if (!(expression))                                                                                                 \
    {                                                                                                                  \
        throw assertion_failure(t_assert_msg(expression, message));                                                    \
    }
#define t_assert_msg(expression, message)                                                                              \
    "Failed assertion " #expression " at " __FILE__ "#" EXPAND_MACRO(__LINE__) ": " message ". "
