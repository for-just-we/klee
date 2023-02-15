//
// Created by prophe on 2023/2/15.
//

#ifndef KLEE_CONTROLLOCATION_H
#define KLEE_CONTROLLOCATION_H

#include "llvm/IR/Instruction.h"


namespace klee {
class ControlLocation {
public:
  std::pair<llvm::Instruction*, llvm::Instruction*> instEdge;
  unsigned int count;

public:
  bool operator<(const ControlLocation& location) const{
    if (instEdge < location.instEdge)
      return true;
    else if (instEdge > location.instEdge)
      return false;
    else
      return count < location.count;
  }

  ControlLocation(std::pair<llvm::Instruction*, llvm::Instruction*> _instEdge, unsigned int _count)
  : instEdge(_instEdge), count(_count) {}
};
}

#endif // KLEE_CONTROLLOCATION_H
