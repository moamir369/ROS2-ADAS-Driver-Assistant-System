#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <iostream>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

class WASDTeleopNode : public rclcpp::Node
{
public:
    WASDTeleopNode() : Node("wasd_teleop_node")

    {
        publisher_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel_ack_stamped", 10);

        RCLCPP_INFO(this->get_logger(), "=== Oyun tarzı (WASD) kontrol düğümü başlatıldı ===");

        RCLCPP_INFO(this->get_logger(), "W : İleri hızlan");

        RCLCPP_INFO(this->get_logger(), "S : Geri git");

        RCLCPP_INFO(this->get_logger(), "A : Sola dön");

        RCLCPP_INFO(this->get_logger(), "D : Sağa dön");

        RCLCPP_INFO(this->get_logger(), "Space (Boşluk tuşu) : Anında dur / Tekerlekleri merkeze getir");

        RCLCPP_INFO(this->get_logger(), "Q / E : Maksimum hızı artır / azalt");

        RCLCPP_INFO(this->get_logger(), "Ctrl+C : Çıkış");
    }

    void run()
    {
        struct termios oldt, newt;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);

        int oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

        char c;
        rclcpp::Rate rate(20); // 20 Hz لضمان سلاسة التحكم والاستجابة

        while (rclcpp::ok())
        {
            if (read(STDIN_FILENO, &c, 1) > 0)
            {
                handle_key(c);
            }

            // إرسال الحركة الحالية بشكل مستمر
            publish_cmd();

            rclcpp::spin_some(this->get_node_base_interface());
            rate.sleep();
        }

        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        fcntl(STDIN_FILENO, F_SETFL, oldf);
    }

private:
    void handle_key(char c)
    {
        switch (c)
        {
        case 'w':
        case 'W':
            current_linear_ = current_speed_;
            break;
        case 's':
        case 'S':
            current_linear_ = -current_speed_;
            break;
        case 'a':
        case 'A':
            current_steering_ = std::min(0.5, current_steering_ + 0.08); // الانعطاف يساراً
            break;
        case 'd':
        case 'D':
            current_steering_ = std::max(-0.5, current_steering_ - 0.08); // الانعطاف يميناً
            break;
        case ' ': // Space - فرامل وإرجاع عجلة القيادة
            current_linear_ = 0.0;
            current_steering_ = 0.0;
            break;
        case 'q':
        case 'Q':
            current_speed_ = std::min(3.0, current_speed_ + 0.2);
            RCLCPP_INFO(this->get_logger(), "السرعة الحالية: %.1f m/s", current_speed_);
            break;
        case 'e':
        case 'E':
            current_speed_ = std::max(0.1, current_speed_ - 0.2);
            RCLCPP_INFO(this->get_logger(), "السرعة الحالية: %.1f m/s", current_speed_);
            break;
        default:
            break;
        }
    }

    void publish_cmd()
    {
        auto msg = geometry_msgs::msg::TwistStamped();
        msg.header.stamp = this->now();
        msg.header.frame_id = "base_link";

        msg.twist.linear.x = current_linear_;
        msg.twist.angular.z = current_steering_;

        publisher_->publish(msg);
    }

    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr publisher_;
    double current_speed_ = 0.8;
    double current_linear_ = 0.0;
    double current_steering_ = 0.0;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WASDTeleopNode>();
    node->run();
    rclcpp::shutdown();
    return 0;
}