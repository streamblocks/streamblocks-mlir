//
// Created by endrix on 25.05.21.
//

#ifndef CALCONTEXT_H
#define CALCONTEXT_H

#include <map>

#include "StreamBlocks/AST/AST.h"

namespace cal {

class CalContext {
public:
  virtual ~CalContext() = default;

  void addNamespace(std::unique_ptr<cal::NamespaceDecl> namespaceDecl) {
    namespaces.push_back(std::move(namespaceDecl));
  }

  void setQID(std::unique_ptr<cal::QID> identifier) {
    qid = std::move(identifier);
  }
  /*
    std::unique_ptr<cal::NamespaceDecl>
    getNamespace(std::unique_ptr<cal::QID> identifier) {
      for (auto &n : namespaces) {
        cal::NamespaceDecl decl = n.get();
        if (*decl.getQID() == *identifier.get()) {
          return n;
        }
      }
    }*/

private:
  std::vector<std::unique_ptr<cal::NamespaceDecl>> namespaces;
  std::unique_ptr<cal::QID> qid;
};

} // namespace cal

#endif // CALCONTEXT_H
