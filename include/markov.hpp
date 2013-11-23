#ifndef __MARKOV_HPP__
#define __MARKOV_HPP__

class Markov
{

public:
  Markov();
  Markov(const Markov& other);
  virtual ~Markov();
  virtual Markov& operator=(const Markov& other);
  virtual bool operator==(const Markov& other) const;
};

#endif
