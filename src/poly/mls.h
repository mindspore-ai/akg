#ifndef _MLS_H
#define _MLS_H

/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// \file mls.h
/// \brief Header file for the MLSched polyhedral scheduler

////////////////////////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////////////////////////

// STL
#include <cstdint>
#include <memory>
#include <vector>
#include <map>

// isl
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

namespace mls {
////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////

namespace runtime {
/// \brief Runtime options for MLSched
class Options;
}  // namespace runtime

/// \brief Scop for MLSched
template <typename T>
class Scop;

/// \brief Influence for MLSched
template <typename T>
class Influence;

/// \brief Hints for MLSched
template <typename T>
class Hints;

namespace bin {
////////////////////////////////////////////////////////////////////////////////
// Version
////////////////////////////////////////////////////////////////////////////////

/// \brief Get the current version major of MLSched
/// \return Current version major of MLSched
long unsigned int VersionMajor(void);

/// \brief Get the current version minor of MLSched
/// \return Current version minor of MLSched
long unsigned int VersionMinor(void);

/// \brief Get the current version patch of MLSched
/// \return Current version patch of MLSched
long unsigned int VersionPatch(void);

/// \brief Get a string representation of the current version of MLSched
/// \return String Representation of the current version of MLSched
std::shared_ptr<char> VersionString(void);

////////////////////////////////////////////////////////////////////////////////
// mls::bin::Options
////////////////////////////////////////////////////////////////////////////////

/// \brief Options for the MLSChed polyhedral scheduler
class Options {
 private:
  /// \brief Inner MLSched options
  mls::runtime::Options *options_{nullptr};

 public:
  ////////////////////////////////////////////////////////////////////////////////
  // Type
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Solver type
  enum class SolverType {
    /// \brief None
    kNone,
    /// \brief Isl solver
    kIsl,
    /// \brief Qiuqi IP solver
    kQiuqiIp,
  };

  /// \brief Get a string representation of a Solver type
  /// \param t Type to represent as a string
  /// \return A string representation of \a t
  /// \relatesalso mls::bin::Options::SolverType
  static std::shared_ptr<char> SolverTypeToString(mls::bin::Options::SolverType t);

  /// \brief Read a SolverType type from a string representation
  /// \param s String to read
  /// \return Read Solver type
  /// \relatesalso mls::bin::Options::SolverType
  [[gnu::nonnull]] static mls::bin::Options::SolverType SolverTypeFromString(const char *str);

  ////////////////////////////////////////////////////////////////////////////////
  // Constructors, destructors, etc.
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Constructor
  Options(void);

  /// \brief Destructor
  ~Options(void);

  /// \brief Copy-Constructor
  /// \param[in] rhs Source mls::bin::Options
  Options(const mls::bin::Options &rhs);

  /// \brief Move-Constructor
  /// \param[in,out] src Source mls::bin::Options
  Options(mls::bin::Options &&src);

  ////////////////////////////////////////////////////////////////////////////////
  // Operators
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Copy-assignment operator
  /// \param[in] rhs Source options
  /// \return Destination options
  mls::bin::Options &operator=(mls::bin::Options rhs);

  /// \brief Dereference operator
  /// \return Pointer to the inner mls::runtime::Options
  mls::runtime::Options *operator*(void);

  /// \brief Dereference operator
  /// \return Pointer to the inner mls::runtime::Options
  [[gnu::const]] const mls::runtime::Options *operator*(void) const;

  ////////////////////////////////////////////////////////////////////////////////
  // Getters
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Get the default verbosity level for MLSched
  /// \return Default verbosity level for MLSched
  static unsigned long int GetDefaultVerbosity(void);

  /// \brief Get the current solver type
  /// \return Current solver type
  [[gnu::pure]] mls::bin::Options::SolverType GetSolverType(void) const;

  /// \brief Get the current verbosity level
  /// \return Current verbosity level
  [[gnu::pure]] unsigned long int GetVerbosity(void) const;

  /// \brief Check whether errors should be logged
  /// \return A boolean value that indicates whether errors should be logged
  /// \retval true if errors should be logged
  /// \retval false otherwise
  [[gnu::pure]] bool ShouldLogErrors(void) const;

  /// \brief Check whether warnings should be logged
  /// \return A boolean value that indicates whether warnings should be logged
  /// \retval true if warnings should be logged
  /// \retval false otherwise
  [[gnu::pure]] bool ShouldLogWarnings(void) const;

  /// \brief Check whether libraries logs should be logged
  /// \return A boolean value that indicates whether libraries logs should be logged
  /// \retval true if libraries logs should be logged
  /// \retval false otherwise
  [[gnu::pure]] bool ShouldLogLibrariesLog(void) const;

  /// \brief Check whether internal debugging should be logged
  /// \return A boolean value that indicates whether internal debugging should be logged
  /// \retval true if internal debugging should be logged
  /// \retval false otherwise
  [[gnu::pure]] bool ShouldLogInternalDebugging(void) const;

  /// \brief Check whether extra internal debugging should be logged
  /// \return A boolean value that indicates whether extra internal debugging should be logged
  /// \retval true if extra internal debugging should be logged
  /// \retval false otherwise
  [[gnu::pure]] bool ShouldLogExtraInternalDebugging(void) const;

  /// \brief Check whether extensive debugging should be logged
  /// \return A boolean value that indicates whether extensive debugging should be logged
  /// \retval true if extensive debugging should be logged
  /// \retval false otherwise
  [[gnu::pure]] bool ShouldLogExtensiveInternalDebugging(void) const;

  /// \brief Get the code sinking behaviour
  /// \return A boolean value that indicates whether the code sinking behaviour is enabled
  /// \retval true if the code sinking behaviour is enabled
  /// \retval false otherwise
  [[gnu::pure]] bool GetCodeSinking(void) const;

  /// \brief Get the constant to parameter behaviour
  /// \return A boolean value that indicates whether the constant to parameter behaviour is enabled
  /// \retval true if the constant to parameter behaviour is enabled
  /// \retval false otherwise
  [[gnu::pure]] bool GetConstantToParameter(void) const;

  /// \brief Get the SCC fusing behaviour
  /// \return A boolean value that indicates whether the SCC maxfuse behaviour is enabled
  /// \retval true if the SCC maxfuse behaviour is enabled
  /// \retval false otherwise
  [[gnu::pure]] bool GetSccMaxfuse(void) const;

  /// \brief Get the parameter shifting behaviour
  /// \return A boolean value that indicates whether parameter shifting is enabled
  /// \retval true if parameter shifting is enabled
  /// \retval false otherwise
  [[gnu::pure]] bool GetParameterShifting(void) const;

  /// \brief Get the matrix init behaviour
  /// \return A boolean value that indicates whether lp problems should be initialized with whole matrices
  /// \retval true if whole matrix initialization is enabled
  /// \retval false otherwise
  [[gnu::pure]] bool GetWholeMatrixInitialization(void) const;

  /// \brief Get the full sets post processing behaviour
  /// \return A boolean value that indicates whether full sets post processing is enabled or disabled
  /// \retval true full sets post processing is enabled
  /// \retval false otherwise
  [[gnu::pure]] bool GetFullSetsPostProcessing(void) const;

  /// \brief Get the extra parallel outer loop post processing behaviour
  /// \return A boolean value that indicates whether extra parallel outer loop post processing is enabled or disabled
  /// \retval true Extra parallel outer loop post processing is enabled
  /// \retval false otherwise
  [[gnu::pure]] bool GetExtraParallelOuterLoopPostProcessing(void) const;

  /// \brief Get a string representation of object
  /// \return A string representation of the object
  std::shared_ptr<char> String(void) const;

  ////////////////////////////////////////////////////////////////////////////////
  // Setters
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Set the current solver type
  /// \param[in] type New solver type
  void SetSolverType(mls::bin::Options::SolverType type);

  /// \brief Set the current verbosity level
  /// \param[in] level New verbosity level
  void SetVerbosity(unsigned long int level);

  /// \brief Choose the code sinking behaviour
  /// \param[in] toggle Enable or disable the code sinking behaviour
  void SetCodeSinking(bool toggle);

  /// \brief Choose the constant to parameter behaviour
  /// \param[in] toggle Enable or disable the constant to parameter behaviour
  void SetConstantToParameter(bool toggle);

  /// \brief Choose the SCC maxfuse behaviour
  /// \param[in] toggle Enable or disable the SCC maxfuse behaviour
  void SetSccMaxfuse(bool toogle);

  /// \brief Choose the parameter shifting behaviour
  /// \param[in] toggle Enable or disable the parameter shifting behaviour
  void SetParameterShifting(bool toogle);

  /// \brief Choose the whole matrix initialization behaviour
  /// \param[in] toggle Enable or disable the whole matrix initialization behaviour
  void SetWholeMatrixInitialization(bool toogle);

  /// \brief Choose the full sets post processing behaviour
  /// \param[in] toggle Enable or disable full sets post processing
  void SetFullSetsPostProcessing(bool toogle);

  /// \brief Choose the extra parallel outer loop post processing behaviour
  /// \param[in] toggle Enable or disable extra parallel outer loop post processing
  void SetExtraParallelOuterLoopPostProcessing(bool toogle);

  ////////////////////////////////////////////////////////////////////////////////
  // Misc. friend functions
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Swap two options
  /// \param[in,out] lhs Left hand side options
  /// \param[in,out] rhs Right hand side options
  friend void swap(mls::bin::Options &lhs, mls::bin::Options &rhs);
};

////////////////////////////////////////////////////////////////////////////////
// mls::bin::InfluenceOperation
////////////////////////////////////////////////////////////////////////////////

/// \brief InfluenceOperation for the MLSched polyhedral scheduler
class InfluenceOperation {
 public:
  /// \brief Type for the mls::bin::InfluenceOperation class
  enum Type {
    /// \brief Undefined type
    kNone,
    /// \brief Division
    kDivision,
    /// \brief Modulo
    kModulo,
  };

 private:
  /// \brief Target statement
  std::shared_ptr<char> statement_{nullptr};
  /// \brief Target dimension
  size_t dimension_{0};
  /// \brief Value for the operation
  long int value_{0};
  /// \brief Type of the operation
  mls::bin::InfluenceOperation::Type type_{mls::bin::InfluenceOperation::Type::kNone};

 public:
  /// \brief Constructor
  InfluenceOperation(void);

  /// \brief Constructor
  /// \param[in] statement Target statement
  /// \param[in] dimensions Target dimension
  /// \param[in] value Value for the operation
  /// \param[in] type Type of the operation
  [[gnu::nonnull]] InfluenceOperation(
    const char *statement, size_t dimension, long int value,
    mls::bin::InfluenceOperation::Type type = mls::bin::InfluenceOperation::Type::kNone);

  /// \brief Get the target statement
  /// \result The target statement
  std::shared_ptr<char> GetStatement(void) const;

  /// \brief Get the target dimension
  /// \result The target dimensions
  size_t GetDimension(void) const;

  /// \brief Get the value for the operation
  /// \result The value for the operation
  long int GetValue(void) const;

  /// \brief Get the type of the operation
  /// \result The value type of operation
  mls::bin::InfluenceOperation::Type GetType(void) const;
};

////////////////////////////////////////////////////////////////////////////////
// mls::bin::Influence
////////////////////////////////////////////////////////////////////////////////

/// \brief Influence for the MLSched polyhedral scheduler
///
/// Influence can be used to provide additional constraints to the
/// polyhedral scheduler and to attempt to influence its behaviour.
class Influence {
 private:
  /// \brief Inner MLSched Influence
  mls::Influence<int64_t> *influence_{nullptr};

 public:
  ////////////////////////////////////////////////////////////////////////////////
  // Constructors, destructors, etc.
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Constructor
  Influence(void);

  /// \brief Destructor
  ~Influence(void);

  /// \brief Copy-Constructor
  /// \param[in] src Source mls::bin::Influence
  Influence(const mls::bin::Influence &src);

  /// \brief Move-Constructor
  /// \param[in,out] src Source mls::bin::Influence
  Influence(mls::bin::Influence &&src);

  /// \brief Constructor from a serialized json string of a MindTrick
  /// \param[in] str Serialized json string of a MindTrick
  [[gnu::nonnull]] Influence(const char *str);

  /// \brief Constructor from a serialized json string of a MindTrick
  /// \param[in] str Serialized json string of a MindTrick
  /// \param[in] options MLSched options
  [[gnu::nonnull]] Influence(const char *str, const mls::bin::Options &options);

  ////////////////////////////////////////////////////////////////////////////////
  // Operators
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Copy-assignment operator
  /// \param[in] rhs Source Influence
  /// \return Destination options
  mls::bin::Influence &operator=(mls::bin::Influence rhs);

  /// \brief Dereference operator
  /// \return Pointer to the inner mls::Influence
  mls::Influence<int64_t> *operator*(void);

  /// \brief Dereference operator
  /// \return Pointer to the inner mls::Influence
  [[gnu::const]] const mls::Influence<int64_t> *operator*(void) const;

  ////////////////////////////////////////////////////////////////////////////////
  // Getters
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Check whether the Influence are empty
  /// \return A boolean value that indicates whether the Influence are empty
  /// \retval true if the Influence are empty (and should not be used)
  /// \retval false otherwise
  bool Empty(void) const;

  /// \brief Get the ordered vector of modulo and divisions operations
  /// \return The vector of modulo and divisions operations
  std::vector<mls::bin::InfluenceOperation> GetOperations(void) const;

  /// \brief Get a string representation of object
  /// \return A string representation of the object
  /// \note The scop's internal options will be used
  std::shared_ptr<char> String(void) const;

  /// \brief Get a string representation of object
  /// \param[in] options Options that may change the string representation
  /// \return A string representation of the object
  std::shared_ptr<char> String(const mls::bin::Options &options) const;

  ////////////////////////////////////////////////////////////////////////////////
  // Setters
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Parse soft constraints from a serialized JSON string
  /// \param[in] str Serialized JSON string
  [[gnu::nonnull]] void ParseSoftConstraints(const char *str);

  ////////////////////////////////////////////////////////////////////////////////
  // Misc. friend functions
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Swap two Influence
  /// \param[in,out] lhs Left hand side Influence
  /// \param[in,out] rhs Right hand side Influence
  friend void swap(mls::bin::Influence &lhs, mls::bin::Influence &rhs);
};

////////////////////////////////////////////////////////////////////////////////
// mls::bin::Hints
////////////////////////////////////////////////////////////////////////////////

/// \brief Hints for the MLSched polyhedral scheduler
///
/// Hints can be used to provide additional constraints to the
/// polyhedral scheduler and to attempt to influence its behaviour.
class Hints {
 private:
  /// \brief Inner MLSched Hints
  mls::Hints<int64_t> *hints_{nullptr};

 public:
  ////////////////////////////////////////////////////////////////////////////////
  // Constructors, destructors, etc.
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Constructor
  Hints(void);

  /// \brief Destructor
  ~Hints(void);

  /// \brief Copy-Constructor
  /// \param[in] src Source mls::bin::Hints
  Hints(const mls::bin::Hints &src);

  /// \brief Move-Constructor
  /// \param[in,out] src Source mls::bin::Hints
  Hints(mls::bin::Hints &&src);

  /// \brief Constructor from a serialized json string of a MindTrick
  /// \param[in] str Serialized json string of a MindTrick
  [[gnu::nonnull]] Hints(const char *str);

  /// \brief Constructor from a serialized json string of a MindTrick
  /// \param[in] str Serialized json string of a MindTrick
  /// \param[in] options MLSched options
  [[gnu::nonnull]] Hints(const char *str, const mls::bin::Options &options);

  /// \brief Constructor from a isl_union_map
  /// \param[in] directives Hints represented as an isl_union_map
  /// \param[in] options Runtime options
  [[gnu::nonnull]] Hints(__isl_keep isl_union_map *const directives, const mls::bin::Options &options);

  ////////////////////////////////////////////////////////////////////////////////
  // Operators
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Copy-assignment operator
  /// \param[in] rhs Source Hints
  /// \return Destination options
  mls::bin::Hints &operator=(mls::bin::Hints rhs);

  /// \brief Dereference operator
  /// \return Pointer to the inner mls::Hints
  mls::Hints<int64_t> *operator*(void);

  /// \brief Dereference operator
  /// \return Pointer to the inner mls::Hints
  [[gnu::const]] const mls::Hints<int64_t> *operator*(void) const;

  ////////////////////////////////////////////////////////////////////////////////
  // Getters
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Check whether the Hints are empty
  /// \return A boolean value that indicates whether the Hints are empty
  /// \retval true if the Hints are empty (and should not be used)
  /// \retval false otherwise
  [[gnu::pure]] bool Empty(void) const;

  /// \brief Check whether the Hints have some directives (serials, vectorials, parallels, etc.)
  /// \return A boolean value that indicates whether the Hints have some directives
  /// \retval true if the Hints have some directives
  /// \retval false otherwise
  [[gnu::pure]] bool HaveDirectives(void) const;

  /// \brief Check whether the Hints has serials directives for a given stateemnt
  /// \param[in] statement Target statement
  /// \return A boolean value that indicates whether the Hints has serials directives for \a statement
  /// \retval true if the Hints has serials directives for \a statement
  /// \retval false otherwise
  [[gnu::pure]] bool HasStatementSerials(const char *statement) const;

  /// \brief Check whether the Hints has vectorials directives for a given stateemnt
  /// \param[in] statement Target statement
  /// \return A boolean value that indicates whether the Hints has vectorials directives for \a statement
  /// \retval true if the Hints has vectorials directives for \a statement
  /// \retval false otherwise
  [[gnu::pure]] bool HasStatementVectorials(const char *statement) const;

  /// \brief Check whether the Hints has reduces directives for a given statement
  /// \param[in] statement Target statement
  /// \return A boolean value that indicates whether the Hints has reduces directives for \a statement
  /// \retval true if the Hints has reduces directives for \a statement
  /// \retval false otherwise
  [[gnu::pure]] bool HasStatementReduces(const char *statement) const;

  /// \brief Check whether the Hints has parallels directives for a given stateemnt
  /// \param[in] statement Target statement
  /// \return A boolean value that indicates whether the Hints has parallels directives for \a statement
  /// \retval true if the Hints has parallels directives for \a statement
  /// \retval false otherwise
  [[gnu::pure]] bool HasStatementParallels(const char *statement) const;

  /// \brief Get the Serials component of the directives for a given statement
  /// \param[in] statement Target statement
  /// \return Serials component of the directives
  [[gnu::pure]] const std::vector<int> &GetStatementSerials(const char *statement) const;

  /// \brief Get the Vectorials component of the directives for a given statement
  /// \param[in] statement Target statement
  /// \return Vectorials component of the directives
  [[gnu::pure]] const std::vector<int> &GetStatementVectorials(const char *statement) const;

  /// \brief Get the Reduces component of the directives for a given statement
  /// \param[in] statement Target statement
  /// \return Reduces component of the directives
  [[gnu::pure]] const std::vector<int> &GetStatementReduces(const char *statement) const;

  /// \brief Get the Parallels component of the directives for a given statement
  /// \param[in] statement Target statement
  /// \return Parallels component of the directives
  [[gnu::pure]] const std::vector<int> &GetStatementParallels(const char *statement) const;

  /// \brief Get the Influence component of the hints
  /// \return Influence component of the hints
  mls::bin::Influence GetInfluence(void) const;

  /// \brief Get a string representation of object
  /// \return A string representation of the object
  /// \note The scop's internal options will be used
  [[gnu::nonnull]] std::shared_ptr<char> String(void) const;

  /// \brief Get a string representation of object
  /// \param[in] options Options that may change the string representation
  /// \return A string representation of the object
  [[gnu::nonnull]] std::shared_ptr<char> String(const mls::bin::Options &options) const;

  ////////////////////////////////////////////////////////////////////////////////
  // Setters
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Set the Serials component of the directives
  /// \param[in] serials Serials component for the directives
  [[gnu::nonnull]] void SetStatementSerials(const char *statement, const std::vector<int> &serials);

  /// \brief Set the Vectorials component of the directives
  /// \param[in] vectorials Vectorials component for the directives
  [[gnu::nonnull]] void SetStatementVectorials(const char *statement, const std::vector<int> &vectorials);

  /// \brief Set the Reduces component of the directives
  /// \param[in] reduces Reduces component for the directives
  [[gnu::nonnull]] void SetStatementReduces(const char *statement, const std::vector<int> &reduces);

  /// \brief Set the Parallels component of the directives
  /// \param[in] parallels Parallels component for the directives
  [[gnu::nonnull]] void SetStatementParallels(const char *statement, const std::vector<int> &parallels);

  /// \brief Set the Influence component of the hints
  /// \param[in] influence Influence component of the hints
  void SetInfluence(const mls::bin::Influence &influence);

  /// \brief Clear all directive components of the Hints
  /// \post GetSerials().empty() == true
  /// \post GetVectorials().empty() == true
  /// \post GetReduces().empty() == true
  /// \post GetParallels().empty() == true
  void ClearDirectives(void);

  /// \brief Parse directives from a serialized JSON string
  /// \param[in] str Serialized json string of a MindTrick
  void ParseDirectives(const char *str);

  /// \brief Parse directives from a serialized JSON string
  /// \param[in] str Serialized json string of a MindTrick
  void ParseInfluence(const char *str);

  ////////////////////////////////////////////////////////////////////////////////
  // Misc. friend functions
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Swap two Hints
  /// \param[in,out] lhs Left hand side Hints
  /// \param[in,out] rhs Right hand side Hints
  friend void swap(mls::bin::Hints &lhs, mls::bin::Hints &rhs);
};

////////////////////////////////////////////////////////////////////////////////
// mls::bin::Scop
////////////////////////////////////////////////////////////////////////////////

/// \brief Scop data for the MLSched polyhedral scheduler
class Scop {
 private:
  /// \brief Pointer to the inner mls::Scop
  mls::Scop<int64_t> *scop_{nullptr};

 public:
  ////////////////////////////////////////////////////////////////////////////////
  // Constructors, destructors, etc.
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Constructor
  Scop(void);

  /// \brief Destructor
  ~Scop(void);

  /// \brief Copy-Constructor
  /// \param[in] src Source mls::bin::Scop
  Scop(const mls::bin::Scop &src);

  /// \brief Move-Constructor
  /// \param[in,out] src Source mls::bin::Scop
  Scop(mls::bin::Scop &&src);

  /// \brief Constructor from isl data
  /// \param[in] sch Initial schedule
  /// \param[in] dependencies Dependencies
  /// \param[in] options Options for MLSched
  /// \param[in] name Optional name for the Scop
  [[gnu::nonnull(1, 2)]] Scop(__isl_keep isl_schedule *sch, __isl_keep isl_union_map *dependencies,
                              const mls::bin::Options &options, const char *name = nullptr);

  /// \brief Constructor from isl data
  /// \param[in] sch Initial schedule
  /// \param[in] dependencies Dependencies
  /// \param[in] reads Reads
  /// \param[in] writes Writes
  /// \param[in] options Options for MLSched
  /// \param[in] name Optional name for the Scop
  [[gnu::nonnull(1, 4)]] Scop(__isl_keep isl_schedule *sch, __isl_keep isl_union_map *dependencies,
                              __isl_keep isl_union_map *reads, __isl_keep isl_union_map *writes,
                              const mls::bin::Options &options, const char *name = nullptr);

  /// \brief Constructor from isl data
  /// \param[in] sch Initial schedule
  /// \param[in] dependencies Dependencies
  /// \param[in] influence Influence for MLSched
  /// \param[in] options Options for MLSched
  /// \param[in] name Optional name for the Scop
  [[gnu::nonnull(1, 2)]] Scop(__isl_keep isl_schedule *sch, __isl_keep isl_union_map *dependencies,
                              const mls::bin::Influence &influence, const mls::bin::Options &options,
                              const char *name = nullptr);

  /// \brief Constructor from isl data
  /// \param[in] sch Initial schedule
  /// \param[in] dependencies Dependencies
  /// \param[in] reads Reads
  /// \param[in] writes Writes
  /// \param[in] influence Influence for MLSched
  /// \param[in] options Options for MLSched
  /// \param[in] name Optional name for the Scop
  [[gnu::nonnull(1, 4)]] Scop(__isl_keep isl_schedule *sch, __isl_keep isl_union_map *dependencies,
                              __isl_keep isl_union_map *reads, __isl_keep isl_union_map *writes,
                              const mls::bin::Influence &influence, const mls::bin::Options &options,
                              const char *name = nullptr);

  /// \brief Constructor from isl data
  /// \param[in] sch Initial schedule
  /// \param[in] dependencies Dependencies
  /// \param[in] hints Hints for MLSched
  /// \param[in] options Options for MLSched
  /// \param[in] name Optional name for the Scop
  [[gnu::nonnull(1, 2)]] Scop(__isl_keep isl_schedule *sch, __isl_keep isl_union_map *dependencies,
                              const mls::bin::Hints &hints, const mls::bin::Options &options,
                              const char *name = nullptr);

  /// \brief Constructor from isl data
  /// \param[in] sch Initial schedule
  /// \param[in] dependencies Dependencies
  /// \param[in] reads Reads
  /// \param[in] writes Writes
  /// \param[in] hints Hints for MLSched
  /// \param[in] options Options for MLSched
  /// \param[in] name Optional name for the Scop
  [[gnu::nonnull(1, 4)]] Scop(__isl_keep isl_schedule *sch, __isl_keep isl_union_map *dependencies,
                              __isl_keep isl_union_map *reads, __isl_keep isl_union_map *writes,
                              const mls::bin::Hints &hints, const mls::bin::Options &options,
                              const char *name = nullptr);

  ////////////////////////////////////////////////////////////////////////////////
  // Operators
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Copy-assignment operator
  /// \param[in] rhs Source Scop
  /// \return Destination Scop
  mls::bin::Scop &operator=(mls::bin::Scop rhs);

  /// \brief Dereference operator
  /// \return Pointer to the inner mls::Scop
  mls::Scop<int64_t> *operator*(void);

  /// \brief Dereference operator
  /// \return Pointer to the inner mls::Hints
  [[gnu::const]] const mls::Scop<int64_t> *operator*(void) const;

  ////////////////////////////////////////////////////////////////////////////////
  // Operations
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Compute a schedule
  /// \pre The Scop contains an initial schedule and dependencies
  [[nodiscard]] bool ComputeSchedule(void);

  ////////////////////////////////////////////////////////////////////////////////
  // Getters
  ////////////////////////////////////////////////////////////////////////////////

  /// \brief Get the computed schedule as an isl schedule tree
  /// \param[in] ctx isl_ctx to use for the conversion into an isl schedule tree
  /// \return The computed schedule represented as an isl schedule tree
  /// \pre ComputeSchedule() has been called and returned true
  /// \seealso mls::bin::Scop::ComputeSchedule()
  [[nodiscard]] __isl_give isl_schedule *ToIslSchedule(__isl_keep isl_ctx *ctx) const;

  /// \brief Check a schedule
  /// \param[in] schedule Schedule to check
  /// \return A boolean value that indicates whether the schedule is valid
  /// \retval true if the schedule is valid
  /// \retval false otherwise
  bool CheckSchedule(__isl_keep isl_schedule *schedule) const;

  /// \brief Get a string representation of object
  /// \return A string representation of the object
  /// \note The scop's internal options will be used
  std::shared_ptr<char> String(void) const;

  /// \brief Get a string representation of object
  /// \param[in] options Options that may change the string representation
  /// \return A string representation of the object
  std::shared_ptr<char> String(const mls::bin::Options &options) const;

  ////////////////////////////////////////////////////////////////////////////////
  // Misc. friend functions
  ////////////////////////////////////////////////////////////////////////////////

  friend void swap(mls::bin::Scop &lhs, mls::bin::Scop &rhs);
};
}  // namespace bin
}  // namespace mls

#endif  // _MLS_H
