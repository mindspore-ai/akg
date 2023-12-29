/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file tvm/node/reflection.h
 * \brief Reflection and serialization of compiler IR/AST nodes.
 */

/*
 * 2023.03.25 - Add TVM 0.8 attributes to the node and conversion pass for exporting TVM 0.8 IR.
 */

#ifndef TVM_NODE_REFLECTION_H_
#define TVM_NODE_REFLECTION_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>

#include <vector>
#include <string>

namespace air {

// forward declaration
class DataType;

using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;

/*!
 * \brief Visitor class for to get the attributesof a AST/IR node.
 *  The content is going to be called for each field.
 *
 *  Each objects that wants reflection will need to implement
 *  a VisitAttrs function and call visitor->Visit on each of its field.
 */
class AttrVisitor {
 public:
  AttrVisitor(std::string tvm_version = tvm06_version) { this->tvm_version = tvm_version; }

  //! \cond Doxygen_Suppress
  TVM_DLL virtual ~AttrVisitor() = default;
  TVM_DLL virtual void Visit(const char* key, double* value) = 0;
  TVM_DLL virtual void Visit(const char* key, int64_t* value) = 0;
  TVM_DLL virtual void Visit(const char* key, uint64_t* value) = 0;
  TVM_DLL virtual void Visit(const char* key, int* value) = 0;
  TVM_DLL virtual void Visit(const char* key, bool* value) = 0;
  TVM_DLL virtual void Visit(const char* key, std::string* value) = 0;
  TVM_DLL virtual void Visit(const char* key, void** value) = 0;
  TVM_DLL virtual void Visit(const char* key, DataType* value) = 0;
  TVM_DLL virtual void Visit(const char* key, runtime::NDArray* value) = 0;
  TVM_DLL virtual void Visit(const char* key, runtime::ObjectRef* value) = 0;
  template <typename ENum, typename = typename std::enable_if<std::is_enum<ENum>::value>::type>
  void Visit(const char* key, ENum* ptr) {
    static_assert(std::is_same<int, typename std::underlying_type<ENum>::type>::value,
                  "declare enum to be enum int to use visitor");
    this->Visit(key, reinterpret_cast<int*>(ptr));
  }

 protected:
  std::string tvm_version;
  //! \endcond
};

/*!
 * \brief Virtual function table to support IR/AST node reflection.
 *
 * Functions are stored  in columar manner.
 * Each column is a vector indexed by Object's type_index.
 */
class ReflectionVTable {
 public:
  template <typename T>
  struct TVMVersionSelector {
    using TVMVersionMap = std::map<std::string, T, std::greater<std::string>>;
  };
  /*!
   * \brief Visitor function.
   * \note We use function pointer, instead of std::function
   *       to reduce the dispatch overhead as field visit
   *       does not need as much customization.
   */
  typedef void (*FVisitAttrs)(Object* self, AttrVisitor* visitor);
  /*!
   * \brief creator function.
   * \param global_key Key that identifies a global single object.
   *        If this is not empty then FGlobalKey must be defined for the object.
   * \return The created function.
   */
  typedef ObjectPtr<Object> (*FCreate)(const std::string& global_key);
  /*!
   * \brief Global key function, only needed by global objects.
   * \param node The node pointer.
   * \return node The global key to the node.
   */
  typedef std::string (*FGlobalKey)(const Object* self);
  /*!
   * \brief Function to get a byte representation that can be used to recover the object.
   * \param node The node pointer.
   * \return bytes The bytes that can be used to recover the object.
   */
  typedef std::string (*FReprBytes)(const Object* self);

  typedef std::string (*FRenameInSpecialCase)(const Object* self);
  /*!
   * \brief Dispatch the VisitAttrs function.
   * \param self The pointer to the object.
   * \param visitor The attribute visitor.
   */
  inline void VisitAttrs(Object* self, AttrVisitor* visitor, const std::string& version) const;
  /*!
   * \brief Get global key of the object, if any.
   * \param self The pointer to the object.
   * \return the global key if object has one, otherwise return empty string.
   */
  inline std::string GetGlobalKey(Object* self) const;
  /*!
   * \brief Get repr bytes if any.
   * \param self The pointer to the object.
   * \param repr_bytes The output repr bytes, can be null, in which case the function
   *                   simply queries if the ReprBytes function exists for the type.
   * \return Whether repr bytes exists.
   */
  inline bool GetReprBytes(const Object* self, std::string* repr_bytes) const;
  /*!
   * \brief Rename the type_key of object.
   * \param self The pointer to the object.
   * \param rename_str The output rename string, can be null, in which case the function
   *                   simply queries if the RenameTypeKey function exists for the type.
   * \return Whether rename str exists.
   */
  inline bool RenameTypeKey(const Object* self, std::string* type_key_str,
                            const std::string& version) const;

  inline bool GetRenameTypeKeyInSpcialCase(const Object* self, std::string* new_type_key, const std::string& version) const;
  /*!
   * \brief Create an initial object using default constructor
   *        by type_key and global key.
   *
   * \param type_key The type key of the object.
   * \param global_key A global key that can be used to uniquely identify the object if any.
   */
  TVM_DLL ObjectPtr<Object> CreateInitObject(const std::string& type_key,
                                             const std::string& global_key = "") const;
  /*!
   * \brief Get an field object by the attr name.
   * \param self The pointer to the object.
   * \param attr_name The name of the field.
   * \return The corresponding attribute value.
   * \note This function will throw an exception if the object does not contain the field.
   */
  TVM_DLL runtime::TVMRetValue GetAttr(Object* self, const std::string& attr_name) const;

  /*!
   * \brief List all the fields in the object.
   * \return All the fields.
   */
  TVM_DLL std::vector<std::string> ListAttrNames(Object* self) const;

  /*! \return The global singleton. */
  TVM_DLL static ReflectionVTable* Global();

  class Registry;
  template <typename T, typename TraitName>
  inline Registry Register();

 private:
  /*! \brief Attribute visitor. */
  std::vector<TVMVersionSelector<FVisitAttrs>::TVMVersionMap> fvisit_attrs_;
  /*! \brief Creation function. */
  std::vector<FCreate> fcreate_;
  /*! \brief Global key function. */
  std::vector<FGlobalKey> fglobal_key_;
  /*! \brief ReprBytes function. */
  std::vector<FReprBytes> frepr_bytes_;
  std::vector<TVMVersionSelector<FRenameInSpecialCase>::TVMVersionMap> frename_type_key_in_spcial_case;
  /*! \brief Rename type_key for different TVM version, default is 0.8.0 */
  std::vector<TVMVersionSelector<std::string>::TVMVersionMap> rename_type_key;
};

/*! \brief Registry of a reflection table. */
class ReflectionVTable::Registry {
 public:
  Registry(ReflectionVTable* parent, uint32_t type_index)
      : parent_(parent), type_index_(type_index) {}
  /*!
   * \brief Set fcreate function.
   * \param f The creator function.
   * \return rference to self.
   */
  Registry& set_creator(FCreate f) {  // NOLINT(*)
    CHECK_LT(type_index_, parent_->fcreate_.size());
    parent_->fcreate_[type_index_] = f;
    return *this;
  }
  /*!
   * \brief Set global_key function.
   * \param f The creator function.
   * \return rference to self.
   */
  Registry& set_global_key(FGlobalKey f) {  // NOLINT(*)
    CHECK_LT(type_index_, parent_->fglobal_key_.size());
    parent_->fglobal_key_[type_index_] = f;
    return *this;
  }
  /*!
   * \brief Set bytes repr function.
   * \param f The ReprBytes function.
   * \return Reference to self.
   */
  Registry& set_repr_bytes(FReprBytes f) {  // NOLINT(*)
    CHECK_LT(type_index_, parent_->frepr_bytes_.size());
    parent_->frepr_bytes_[type_index_] = f;
    return *this;
  }
  /*!
   * \brief Set the type_key for different TVM version during serialization, default is 0.8.0
   * \param s The new type_key.
   * \param version The tvm version corresponding to the type_key.
   * \return Reference to self.
   */
  Registry& set_rename_type_key(std::string type_key, std::string version = tvm08_version) {
    CHECK_LT(type_index_, parent_->rename_type_key.size());
    auto current_map = parent_->rename_type_key[type_index_];
    std::pair item(version, type_key);
    current_map.emplace(item);
    parent_->rename_type_key[type_index_] = current_map;
    return *this;
  }
  Registry& set_rename_type_key_in_special_case(FRenameInSpecialCase f, std::string version = tvm08_version) {
    CHECK_LT(type_index_, parent_->frename_type_key_in_spcial_case.size());
    std::pair item(version, f);
    parent_->frename_type_key_in_spcial_case[type_index_].emplace(item);
    return *this;
  }
 private:
  ReflectionVTable* parent_;
  uint32_t type_index_;
};

#define TVM_REFLECTION_REG_VAR_DEF \
  static TVM_ATTRIBUTE_UNUSED ::air::ReflectionVTable::Registry __make_reflection

/*!
 * \brief Directly register reflection VTable.
 * \param TypeName The name of the type.
 * \param TraitName A trait class that implements functions like VisitAttrs and SEqualReduce.
 *
 * \note This macro can be called in different place as TVM_REGISTER_OBJECT_TYPE.
 *       And can be used to register the related reflection functions for runtime objects.
 */
#define TVM_REGISTER_REFLECTION_VTABLE(TypeName, TraitName) \
  TVM_STR_CONCAT(TVM_REFLECTION_REG_VAR_DEF, __COUNTER__) = \
      ::air::ReflectionVTable::Global()->Register<TypeName, TraitName>()

/*!
 * \brief Register a node type to object registry and reflection registry.
 * \param TypeName The name of the type.
 * \note This macro will call TVM_REGISTER_OBJECT_TYPE for the type as well.
 */
#define TVM_REGISTER_NODE_TYPE(TypeName)                                             \
  TVM_REGISTER_OBJECT_TYPE(TypeName);                                                \
  TVM_REGISTER_REFLECTION_VTABLE(TypeName, ::air::detail::ReflectionTrait<TypeName>) \
      .set_creator([](const std::string&) -> ObjectPtr<Object> {                     \
        return ::air::runtime::make_object<TypeName>();                              \
      })

// Implementation details
namespace detail {

template <typename T>
struct ImplVisitAttrs {
  static void VisitAttrs(T* self, AttrVisitor* v) { self->VisitAttrs(v); }
};

template <typename T>
struct ImplVisitAttrsForTVM08 {
  static void VisitAttrsForTVM08(T* self, AttrVisitor* v) { self->VisitAttrsForTVM08(v); }
};

template <typename T>
struct ReflectionTrait : public ImplVisitAttrs<T>, public ImplVisitAttrsForTVM08<T> {};

template <typename T, typename TraitName>
struct SelectVisitAttrs {
  static void VisitAttrs(Object* self, AttrVisitor* v) {
    TraitName::VisitAttrs(static_cast<T*>(self), v);
  }
};

template <typename T, typename TraitName, bool = T::_type_has_method_visit_attrs_for_tvm08>
struct SelectVisitAttrsForTVM08 : public SelectVisitAttrs<T, TraitName> {
  static constexpr const auto VisitAttrsForTVM08 = SelectVisitAttrs<T, TraitName>::VisitAttrs;
};

template <typename T, typename TraitName>
struct SelectVisitAttrsForTVM08<T, TraitName, true> {
  static void VisitAttrsForTVM08(Object* self, AttrVisitor* v) {
    TraitName::VisitAttrsForTVM08(static_cast<T*>(self), v);
  }
};

}  // namespace detail

// Implementation details
template <typename T, typename TraitName>
inline ReflectionVTable::Registry ReflectionVTable::Register() {
  uint32_t tindex = T::RuntimeTypeIndex();
  if (tindex >= fvisit_attrs_.size()) {
    fvisit_attrs_.resize(tindex + 1, {});
    fcreate_.resize(tindex + 1, nullptr);
    frepr_bytes_.resize(tindex + 1, nullptr);
    rename_type_key.resize(tindex + 1, {});
    frename_type_key_in_spcial_case.resize(tindex + 1, {});
    fglobal_key_.resize(tindex + 1, nullptr);
  }
  fvisit_attrs_[tindex][tvm06_version] = ::air::detail::SelectVisitAttrs<T, TraitName>::VisitAttrs;
  fvisit_attrs_[tindex][tvm08_version] = ::air::detail::SelectVisitAttrsForTVM08<T, TraitName>::VisitAttrsForTVM08;

  return Registry(this, tindex);
}

inline void ReflectionVTable::VisitAttrs(Object* self, AttrVisitor* visitor,
                                         const std::string& version) const {
  uint32_t tindex = self->type_index();
  if (tindex >= fvisit_attrs_.size() || fvisit_attrs_[tindex].empty()) {
    LOG(FATAL) << "TypeError: " << self->GetTypeKey()
               << " is not registered via TVM_REGISTER_NODE_TYPE";
  }
  auto visit_map = fvisit_attrs_[tindex];
  for (auto& item : visit_map) {
    // Find the first key less than or equal to the current version number
    if (item.first <= version) {
      auto func = item.second;
      func(self, visitor);
      return;
    }
  }
  LOG(FATAL) << "TypeError: " << self->GetTypeKey()
             << " does not have VisitAttrs function for TVM version " << version
             << ", only sypport version larger than :" << tvm06_version;
}

inline std::string ReflectionVTable::GetGlobalKey(Object* self) const {
  uint32_t tindex = self->type_index();
  if (tindex < fglobal_key_.size() && fglobal_key_[tindex] != nullptr) {
    return fglobal_key_[tindex](self);
  } else {
    return std::string();
  }
}

inline bool ReflectionVTable::GetReprBytes(const Object* self, std::string* repr_bytes) const {
  uint32_t tindex = self->type_index();
  if (tindex < frepr_bytes_.size() && frepr_bytes_[tindex] != nullptr) {
    if (repr_bytes != nullptr) {
      *repr_bytes = frepr_bytes_[tindex](self);
    }
    return true;
  } else {
    return false;
  }
}

inline bool ReflectionVTable::RenameTypeKey(const Object* self, std::string* type_key_str,
                                            const std::string& version) const {
  uint32_t tindex = self->type_index();
  if (tindex < rename_type_key.size() && !rename_type_key[tindex].empty()) {
    if (type_key_str != nullptr) {
      auto current_map = rename_type_key[tindex];
      for (auto& item : current_map) {
        // Find the first key less than or equal to the current version number
        if (item.first <= version) {
          *type_key_str = item.second;
          return true;
        }
      }
    }
    return false;
  } else {
    return false;
  }
}

inline bool ReflectionVTable::GetRenameTypeKeyInSpcialCase(const Object* self, std::string* new_type_key, const std::string& version) const {
  uint32_t tindex = self->type_index();
  if (tindex < frename_type_key_in_spcial_case.size() && !frename_type_key_in_spcial_case[tindex].empty()) {
    if (new_type_key != nullptr) {
      auto current_map = frename_type_key_in_spcial_case[tindex];
      for (auto& item : current_map) {
        // Find the first key less than or equal to the current version number
        if (item.first <= version) {
          *new_type_key = item.second(self);
          return true;
        }
      }
    }
    return false;
  } else {
    return false;
  }
}
}  // namespace air
#endif  // TVM_NODE_REFLECTION_H_
