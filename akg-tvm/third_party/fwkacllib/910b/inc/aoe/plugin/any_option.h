/**
 * @file any_option.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef AOE_PLUGIN_ANY_OPTION_H
#define AOE_PLUGIN_ANY_OPTION_H

#include <typeindex>
#include <memory>

namespace Aoe {
class AnyOption {
public:
    AnyOption() : placeholder_(nullptr), index_(typeid(void)) {}

    template <class T,
        class = typename std::enable_if<!std::is_same<typename std::decay<T>::type, AnyOption>::value, T>::type>
    explicit AnyOption(T &&rhs)
        : placeholder_(new (std::nothrow) Holder<typename std::decay<T>::type>(std::forward<T>(rhs))),
          index_(typeid(typename std::decay<T>::type))
    {}

    AnyOption(const AnyOption &rhs) : placeholder_(rhs.Clone()), index_(rhs.index_) {}

    AnyOption(AnyOption &&rhs) : placeholder_(std::move(rhs.placeholder_)), index_(std::move(rhs.index_)) {}

    AnyOption &operator = (const AnyOption &rhs)
    {
        if (this == &rhs) {
            return *this;
        }

        placeholder_ = rhs.Clone();
        index_ = rhs.index_;
        return *this;
    }

    AnyOption &operator = (AnyOption &&rhs)
    {
        if (this == &rhs) {
            return *this;
        }

        placeholder_ = std::move(rhs.placeholder_);
        index_ = std::move(rhs.index_);
        return *this;
    }

    bool Valid() const
    {
        return placeholder_ != nullptr;
    }

    template <typename T> bool IsType() const;

    template <typename T> bool Get(T &value) const;

    template <typename T> T *Get() const;

private:
    class Placeholder {
    public:
        virtual ~Placeholder() = default;
        virtual std::unique_ptr<Placeholder> Clone() const = 0;
    };

    template <typename T> class Holder : public Placeholder {
    public:
        template <typename... Args> explicit Holder(Args &&... args) : value_(std::forward<Args>(args)...) {}

        std::unique_ptr<Placeholder> Clone() const override
        {
            return std::unique_ptr<Placeholder>(new (std::nothrow) Holder(value_));
        }

        friend class AnyOption;

    private:
        T value_;
    };

    std::unique_ptr<Placeholder> Clone() const
    {
        if (placeholder_) {
            return placeholder_->Clone();
        }

        return nullptr;
    }

    std::unique_ptr<Placeholder> placeholder_;
    std::type_index index_;
};

template <typename T> bool AnyOption::IsType() const
{
    return index_ == std::type_index(typeid(T));
}

template <typename T> bool AnyOption::Get(T &value) const
{
    T *tp = Get<T>();
    if (tp != nullptr) {
        value = *tp;
        return true;
    }
    return false;
}

template <typename T> T *AnyOption::Get() const
{
    if (Valid() && IsType<T>()) {
        auto holder = dynamic_cast<Holder<T> *>(placeholder_.get());
        return &holder->value_;
    }
    return nullptr;
}
} // namespace Aoe
#endif // AOE_PLUGIN_ANY_OPTION_H