#include "StreamBlocks/AST/AST.h"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace cal;

namespace {

// RAII helper to manage increasing/decreasing the indentation as we traverse
// the AST
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class ASTDumper {
public:
  void dump(NamespaceDecl *decl);

private:
  void dump(Expression *expr);
  void dump(Statement *stmt);
  void dump(Import *import);
  void dump(SingleImport *import);
  void dump(GroupImport *import);
  void dump(Decl *decl);
  void dump(NLNetwork *network);
  void dump(CalActor *actor);
  void dump(Action *action);

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int i = 0; i < curIndent; i++)
      llvm::errs() << "  ";
  }
  int curIndent = 0;
};

} // namespace

/// Return a formatted string for the location of any node
template <typename T> static std::string loc(T *node) {
  const auto &loc = node->loc();
  return (llvm::Twine("@") + *loc.begin.filename + ":" +
          llvm::Twine(loc.begin.line) + ":" + llvm::Twine(loc.begin.column))
      .str();
}

// Helper Macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

/// Print an Import
void ASTDumper::dump(Import *import) {
  llvm::TypeSwitch<Import *>(import)
      .Case<SingleImport, GroupImport>([&](auto *node) { this->dump(node); })
      .Default([&](Import *) {
        // No match, fallback to a generic message
        INDENT();
        llvm::errs() << "<unknown Import, kind " << import->getKind() << ">\n";
      });
}

void ASTDumper::dump(SingleImport *import) {
  INDENT();
  llvm::errs() << "Single Import"
               << "\n";
  indent();
  if (import->getLocalName().empty()) {
    llvm::errs() << "Kind : " << import->getPrefix()
                 << ", QID: " << import->getGlobalName()->toString() << "\n";
  } else {
    llvm::errs() << "Kind : " << import->getPrefix()
                 << ", QID: " << import->getGlobalName()->toString()
                 << ", LocalName :" << import->getLocalName() << "\n";
  }
}

void ASTDumper::dump(GroupImport *import) {
  INDENT();
  llvm::errs() << "Group Import" << loc(import) << "\n";
  indent();
  llvm::errs() << "Kind : " << import->getPrefix()
               << ", QID: " << import->getGlobalName()->toString() << "\n";
}

/// Print a Namespace
void ASTDumper::dump(NamespaceDecl *decl) {
  INDENT();
  if (decl->getQID() == nullptr) {
    llvm::errs() << "Namespace: "
                 << "(default namespace)"
                 << "\n";
  } else {
    llvm::errs() << "Namespace: " << decl->getQID()->toString() << "\n";
  }

  // Imports
  for (auto &import : decl->getImports()) {
    dump(import.get());
  }

  // Global Variable declarations
  for (auto &varDecl : decl->getvarDecls()) {
    // TODO
  }

  // Global Entity declarations
  for (auto &entity : decl->getEntityDecls()) {
    // TODO
  }
}

namespace cal {
void dump(NamespaceDecl &namespaceDecl) { ASTDumper().dump(&namespaceDecl); }
} // namespace cal