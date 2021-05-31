#include <iostream>

#include "StreamBlocks/AST/AST.h"

#include "driver.h"

int
main (int argc, char *argv[])
{
  cal::location loc;
  std::vector<std::unique_ptr<cal::TypeExpr>> fTypeParams;
  //std::unique_ptr<cal::TypeExpr> functionReturnType = std::make_unique<cal::NominalTypeExpr>(loc, "int", );

  std::unique_ptr<cal::Expression> a = std::make_unique<cal::ExprVariable>(loc, "titi");


  std::unique_ptr<cal::Expression> cc = a->clone();



  int res = 0;
  driver drv;
  for (int i = 1; i < argc; ++i)
    if (argv[i] == std::string ("-p"))
      drv.trace_parsing = true;
    else if (argv[i] == std::string ("-s"))
      drv.trace_scanning = true;
    else if (!drv.parse (argv[i]))
      std::cout << drv.result << '\n';
    else
      res = 1;
  return res;
}
