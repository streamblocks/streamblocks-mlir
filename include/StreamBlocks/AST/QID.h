//
// Created by endrix on 19.05.21.
//

#ifndef QID_H
#define QID_H

#include <string>
#include <vector>

namespace cal {

class QID {
public:
  QID(std::vector<std::string> parts) : parts(std::move(parts)) {}
  virtual ~QID() {}

  std::unique_ptr<QID> concat(QID &qid) {
    std::vector<std::string> elements = parts;
    for (auto i : qid.parts) {
      elements.push_back(i);
    }
    return std::make_unique<QID>(elements);
  }

  /**
   * Returns the number of names in this QID.
   *
   * @return the number of names in this QID
   */
  int getNameCount() { return parts.size(); }

  /**
   * Returns a QID consisting of the first name of this QID.
   *
   * @return the first name of this QID or null if this QID is empty
   */
  std::unique_ptr<QID> getFirst() { return part(0, 1); }

  /**
   * Returns a QID consisting of the last name of this QID.
   *
   * @return the last name of this QID or null if this QID is empty
   */
  std::unique_ptr<QID> getLast() {
    int count = getNameCount();
    return part(count - 1, count);
  }

  /**
   * Returns a QID consisting of the sequence of names in this QID except the
   * last name.
   *
   * @return the sequence of names except the last name in this QID or null if
   *         this QID is empty
   */
  std::unique_ptr<QID> getButLast() {
    int count = getNameCount();
    return part(0, count - 1);
  }

  /**
   * Parses the given name into a QID by splitting the name in each occurrence
   * of '.'. The empty string yields the empty QID, but in all other cases are
   * empty names illegal.
   *
   * @param name a dot-separated qualified name
   * @return a QID from the given name
   *
   */
  static std::unique_ptr<QID> parse(std::string name) {
    if (name.empty()) {
      return empty();
    }
    std::stringstream ss(name);
    std::string item;
    std::vector<std::string> parts;
    while (std::getline(ss, item, '.')) {
      if (item.length() > 0) {
        parts.push_back(item);
      }
    }
    return std::make_unique<QID>(parts);
  }

  /**
   * Returns an empty QID.
   *
   * @return an empty QID
   */
  static std::unique_ptr<QID> empty() {
    return std::make_unique<QID>(std::vector<std::string>());
  }

  /**
   * Returns true if the first names of the given QID are equal to the names
   * of this QID. Some examples: "a.b" is a prefix of "a.b.c" and a prefix of
   * "a.b", but "a.b" is not a prefix of "a.bb", "a" or "b".
   *
   * @param that
   *            the QID to compare prefix with
   * @return true if the given QID is a prefix of this QID
   */
  bool isPrefixOf(QID &that) {
    if (this->getNameCount() > that.getNameCount()) {
      return false;
    } else {
      for (int i = 0; i < this->getNameCount(); i++) {
        if (this->parts[i].compare(that.parts[i]) != 0) {
          return false;
        }
      }
      return true;
    }
  }

  bool operator == (const QID& that)
  {
    bool result = std::equal(parts.begin(), parts.end(), that.parts.begin());
    return result;
  }

private:
  std::unique_ptr<QID> part(int from, int to) {
    int count = getNameCount();

    if (from < 0 || from > count || to < from || to > count) {
      return nullptr;
    } else {
      auto first = parts.begin() + from;
      auto last = parts.begin() + to;
      return std::make_unique<QID>(std::vector<std::string>(first, last));
    }
  }

  std::vector<std::string> parts;
};

} // namespace cal

#endif // QID_H
